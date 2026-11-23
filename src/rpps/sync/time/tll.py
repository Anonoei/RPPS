"""Timing locked loop: a per-symbol loop around a timing error detector (rpps.sync.time.ted)"""
from types import SimpleNamespace

import numpy as np

from ...base import Stage
from ..pll import PILoop
from .ted import DETECTORS
from ... import _effort
from ...math import interp

class TLL(Stage):
    """Symbol timing by a timing locked loop: samples at `sps` in (after their matched filter), one symbol each out

        symbols = matched + rp.sync.time.TLL(sps=4, bw=0.01)

    Second class (rpps.sync): a loop updated once per symbol. Each symbol is interpolated at its
    instant, and a timing error detector (`ted`: "zx", zero crossing (Gardner), on the symbol before it, the sample half
    way and this one; "el", early-late, on the samples half a symbol either side of this one) gives the timing error, which a proportional-integral loop filter (rpps.sync.pll.PILoop, from `bw` as a fraction of the
    symbol rate) turns into the next instant. Needs no decisions and no carrier, so it runs before carrier recovery.
    Any sps over 1, fractional too, and any block cut. Estimates in stream.est[name]: the timing phase (samples) and the
    loop's rate correction.
    """
    CONTROLS = {}

    def __init__(self, sps=None, bw=0.01, damping=0.7071, name="timing", ted="zx"):
        if ted not in DETECTORS:
            raise ValueError(f"TLL: ted {ted!r}, one of {sorted(DETECTORS)}")
        self.sps, self.bw, self.damping, self.name, self.ted = sps, bw, damping, name, ted

    def __str__(self):
        return "TLL" if self.ted == "zx" else f"TLL({self.ted})"

    def new_state(self, meta):
        # buf holds input from absolute position `base`; t: the next symbol's instant (absolute, in input samples)
        return SimpleNamespace(buf=np.zeros(0, np.complex64), base=meta.pos, t=None, prev=None,
                               loop=PILoop.from_bandwidth(self.bw, self.damping, Kd=2.0))

    def step(self, x, st, meta):
        sps = float(self.sps or meta.sps)
        if sps <= 1:
            raise ValueError(f"TLL needs more than 1 sample per symbol, got {sps}")
        st.buf = np.concatenate((st.buf, np.asarray(x, dtype=np.complex64)))
        end = st.base + len(st.buf)
        if st.t is None:
            st.t = float(st.base + 1 + sps/2) # (room for the half-symbol sample before it)
        out = []
        # A symbol needs its instant and the half symbol before it, and the cubic its neighbours: up to t + 2
        R, taps = interp.REACH, _effort.interp_taps(sps)
        ahead = sps/2 if self.ted == "el" else 0.0 # (early-late looks half a symbol past the instant)
        while st.t + ahead + (2 if meta.flush else R) < end: # (flushed: zeros after the end, as before the start)
            mid_t = st.t - sps/2
            if mid_t - 1 < st.base:
                st.t += sps
                continue
            if self.ted == "el":
                z, mid, after = interp.fractional_edge(st.buf, np.array([st.t, mid_t, st.t + sps/2]) - st.base, taps)
                st.t += sps*st.loop.run(float(DETECTORS["el"](mid, z, after)))
            else:
                z, mid = interp.fractional_edge(st.buf, np.array([st.t, mid_t]) - st.base, taps)
                if st.prev is not None: # early: the symbol before, late: this one
                    st.t += sps*st.loop.run(float(DETECTORS["zx"](st.prev, mid, z)))
            st.prev = z
            out.append(z)
            st.t += sps
        keep = int(np.floor(st.t - sps/2)) - interp.REACH # what the next symbol's half-way sample needs
        keep = min(max(keep, st.base), end)
        st.buf, st.base = st.buf[keep - st.base:], keep
        meta.stream.est[self.name] = dict(phase=st.t % sps, rate=st.loop.integ)
        meta.op.Fs = meta.Fs/sps
        meta.op.sps = 1
        return np.asarray(out, dtype=np.complex64)
