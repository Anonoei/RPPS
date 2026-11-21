"""Known timing: the symbol instants given, not estimated (a profile's `timing: known`)"""
from types import SimpleNamespace

import numpy as np

from ... import _accel
from ...base import Stage
from ...signal.filters.convolve import convolve

class Known(Stage):
    """Timing that is known, not estimated (a loopback, a source whose timing is given): one sample per symbol at the pulse
    peaks, which the group delay so far (meta.delay) places. Tracks the absolute input index of the next peak, so any block
    cut gives the same output.

    With a matched filter fused in (`fuse`, which a profile does for the filter just before it) the filter is evaluated only at
    those instants, a dot product of its taps with the signal per symbol: `sps` times less work than filtering every sample and
    keeping every `sps`-th, the same symbols. The estimators that need the filter's output at every sample (Oerder & Meyr) have
    nothing to fuse

    anchor: the name of a preamble (rpps.sync.Preamble) in the stream's estimates: the grid starts at its first symbol's
    peak, moved by the group delay added since (a burst's own timing; until the preamble is found nothing comes out)"""
    def __init__(self, sps, anchor=None):
        self.sps = sps
        self.anchor = anchor
        self.matched = None

    def fuse(self, matched):
        """Take over a matched filter (a FIR stage): this stage then also filters, at the instants it samples"""
        self.matched = matched

    def __str__(self):
        return "Known" if self.matched is None else "Known(matched)"

    def new_state(self, meta):
        taps = None if self.matched is None else self.matched.new_state(meta).taps
        return SimpleNamespace(next=None, taps=taps, hist=None)

    def step(self, samples, st, meta):
        sps = int(self.sps or meta.isps)
        delay = meta.delay + (0.0 if st.taps is None else (len(st.taps) - 1)/2/meta.Fs) # (the filter's group delay)
        if st.next is None and self.anchor is not None:
            est = meta.stream.est.get(self.anchor) or {}
            if not est.get("found"):
                meta.op.Fs, meta.op.sps = meta.Fs/sps, 1
                return samples[:0]
            r = meta.Fs/est["Fs"]
            first = int(round(est["first"]*r + (delay - est["delay"])*meta.Fs)) # where the preamble put it, delayed since
            st.next = first if first >= meta.pos else first + -(-(meta.pos - first)//sps)*sps
        if st.next is None:
            first = int(round(delay*meta.Fs)) # absolute index of the first symbol's peak
            st.next = first if first >= meta.pos else first + -(-(meta.pos - first)//sps)*sps
        first_idx = st.next - meta.pos
        count = max(0, -(-(len(samples) - first_idx)//sps))
        st.next += count*sps
        meta.op.Fs = meta.Fs/sps
        meta.op.sps = 1
        if st.taps is None:
            return samples[np.arange(first_idx, len(samples), sps)]
        meta.delay = delay
        return self._matched(samples, st, first_idx, sps, count)

    @staticmethod
    def _matched(samples, st, first_idx, sps, count):
        nh = len(st.taps)
        keep = nh - 1
        x = np.asarray(samples)
        hist = st.hist if st.hist is not None else np.zeros(keep, dtype=x.dtype)
        buf = np.concatenate((hist, x))
        st.hist = buf[len(buf) - keep:]
        if not count:
            return np.zeros(0, np.complex64)
        if _accel.native is not None:
            return _accel.native.fir_decimate(np.ascontiguousarray(buf, dtype=np.complex64), st.taps, first_idx + keep, sps, count)
        return convolve(buf, st.taps, "valid")[first_idx + sps*np.arange(count)].astype(np.complex64)
