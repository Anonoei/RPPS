"""Resampling by any ratio with a windowed-sinc polyphase filter, and an optional frequency shift fused in front of it"""
from types import SimpleNamespace

import numpy as np

from ... import _accel
from .sample import Sample

_NATIVE = (np.float32, np.complex64) # What rpps._native.polyphase_read takes
_CHUNK = 1 << 15 # Outputs gathered at a time by the Python reference: bounds its temporary to _CHUNK x taps

def design_bank(taps, phases, cutoff, beta=7.0):
    """(phases + 1, taps) table of the interpolation filter: row p is the kernel for a read position p/phases of the way past a sample

    taps: even length (the kernel spans taps/2 samples either side of the read position). cutoff: [cycles per input sample] 0..0.5.
    Kaiser window, every row normalised to unit gain at DC so a constant keeps its level wherever it is read"""
    if taps < 2 or taps % 2:
        raise ValueError("taps must be even and at least 2")
    if not 0 < cutoff <= 0.5:
        raise ValueError("cutoff must be in (0, 0.5] cycles per input sample")
    half = taps//2
    f = np.arange(phases + 1)/phases
    d = f[:, None] - np.arange(-half + 1, half + 1)[None, :] # distance from the read position to each tap's sample
    win = np.where(np.abs(d) < half, np.i0(beta*np.sqrt(np.clip(1 - (d/half)**2, 0, 1)))/np.i0(beta), 0.0) # zero at the ends: the last row (a whole sample) is then the first row one sample on
    g = 2*cutoff*np.sinc(2*cutoff*d)*win
    return np.ascontiguousarray(g/g.sum(axis=1, keepdims=True))

def read_python(buf, table, pos, step, nout):
    """The reference for `native.polyphase_read`: output k is `buf` read at pos + k*step, the nearest of the table's phases. A chunk of outputs at a time"""
    half = table.shape[1]//2
    nph = table.shape[0] - 1
    out = np.empty(nout, dtype=np.result_type(buf.dtype, np.float32))
    taps = np.arange(-half + 1, half + 1)
    for a in range(0, nout, _CHUNK):
        p = pos + step*np.arange(a, min(nout, a + _CHUNK))
        i0 = p.astype(np.int64)
        ph = np.floor((p - i0)*nph + 0.5).astype(np.int64)
        out[a:a + len(p)] = np.einsum("nj,nj->n", buf[i0[:, None] + taps[None, :]], table[ph])
    return out

class FractionalPolyphase(Sample):
    """Resample by any ratio (output rate over input rate) with a Kaiser-windowed sinc polyphase filter, and optionally move the spectrum first

        stage = FractionalPolyphase(10.7622/20)                              # 20 MS/s to 10.7622 MS/s, anti-aliased
        stage = FractionalPolyphase(10.7622/20, cutoff=3.7e6, shift=-0.5e6)  # the channel at +0.5 MHz to baseband, then resample

    Output m is the input at time m/ratio, with no delay (as `resample_poly`). The read position and the oscillator phase carry across
    blocks, so any block length gives the same stream, and `ratio` may be changed between blocks (a clock error trim): the position carries
    in input samples and the filter stays as designed. Real input gives real output unless a shift is asked for.

    ratio: output rate / input rate. cutoff: [Hz, at the input rate] edge of the pass band; None: the lower Nyquist, so decimating
        does not alias. taps: filter length (even): it sets the transition width. phases: kernels tabulated over one input sample: the read
        position is rounded to the nearest, so the interpolation error of the position is about 1/(2*phases) of a sample (the
        image it makes is -20*log10(pi*bandwidth/phases) dB below the signal). beta: Kaiser window, sets the stop band depth (7: about 70 dB).
    shift: [Hz] translation applied before the filter, equal to `ShiftFreq(shift)` followed by this stage; the centre frequency moves with it.
    """
    __slots__ = ("ratio", "cutoff", "taps", "phases", "beta", "shift")
    def __init__(self, ratio, cutoff=None, taps=32, phases=512, beta=7.0, shift=0.0):
        if not ratio > 0:
            raise ValueError("ratio must be positive")
        design_bank(taps, 1, 0.25, beta) # Check the settings now
        self.ratio, self.cutoff, self.taps, self.phases, self.beta, self.shift = float(ratio), cutoff, int(taps), int(phases), float(beta), float(shift)

    def bank(self, Fs):
        """The (phases + 1, taps) table for an input sample rate `Fs`"""
        cut = 0.5*min(1.0, self.ratio) if self.cutoff is None else self.cutoff/Fs
        return design_bank(self.taps, self.phases, cut, self.beta)

    def new_state(self, meta):
        table = self.bank(meta.Fs)
        # scratch: the buffer tail + block is built in; tail: the input kept from the last block (mixed), starting as zeros so the first output is the first input sample; pos: where the next output reads
        return SimpleNamespace(table=table, table32=table.astype(np.float32), tail=None, pos=float(self.taps//2 - 1), phase=0.0, scratch=None)

    def step(self, samples, st, meta):
        x = np.asarray(samples)
        meta.op.Fs *= self.ratio
        meta.op.sps *= self.ratio
        meta.op.cf += self.shift
        dt = np.result_type(x.dtype, np.complex64) if self.shift else x.dtype
        if dt.kind not in "fc":
            dt = np.dtype(np.float32) if dt.kind in "biu" else dt
            x = x.astype(dt)
        half = self.taps//2
        if st.tail is None:
            st.tail = np.zeros(half - 1, dtype=dt)
        n = len(x)
        total = len(st.tail) + n
        if st.scratch is None or len(st.scratch) < total or st.scratch.dtype != dt:
            st.scratch = np.empty(total + total//4, dtype=dt) # kept between blocks: a fresh buffer of a block's size costs page faults on every one
        buf = st.scratch[:total]
        buf[:len(st.tail)] = st.tail
        native = _accel.native is not None and dt in _NATIVE
        if self.shift:
            w = 2*np.pi*self.shift/meta.Fs
            if native and n:
                st.phase = _accel.native.rotate_into(np.ascontiguousarray(x, dtype=np.complex64), buf[len(st.tail):], st.phase, w, 0)
            else:
                buf[len(st.tail):] = x*np.exp(1j*(st.phase + w*np.arange(n)))
                st.phase = (st.phase + w*n) % (2*np.pi)
        else:
            buf[len(st.tail):] = x
        step = 1/self.ratio
        nout = int(np.floor((len(buf) - 1 - half - st.pos)/step)) + 1 # Outputs whose kernel is wholly inside the buffer
        if nout <= 0:
            st.tail = buf.copy()
            return np.zeros(0, dtype=dt)
        if native:
            out = _accel.native.polyphase_read(buf, st.table32, st.pos, step, nout, 0)
        else:
            out = read_python(buf, st.table, st.pos, step, nout).astype(dt, copy=False)
        nxt = st.pos + step*nout
        keep = max(0, int(nxt) - half + 1)
        st.tail = buf[keep:].copy()
        st.pos = nxt - keep
        return out

def fractional_resample(x, ratio, Fs=1.0, **kw):
    """Resample the array `x` (sample rate `Fs`, which only matters to `cutoff` and `shift`, in Hz) by `ratio`: a one-call `FractionalPolyphase`"""
    from ...base import Data
    return np.asarray((Data(x, Fs=Fs, sps=1, flush=True) + FractionalPolyphase(ratio, **kw)).data)
