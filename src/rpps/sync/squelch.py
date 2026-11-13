"""Squelch: detects that a signal is present, and gates the stream on it"""
from types import SimpleNamespace

import numpy as np

from .. import _accel
from ..base import Stage

_NATIVE_DTYPES = (np.float32, np.float64, np.complex64, np.complex128) # What the rpps._native loops take

class Squelch(Stage):
    """Silence the signal while it is quieter than `threshold_db` (power, dB re 1.0), with hysteresis

    The level follows a running average of the power, so a brief dip doesn't close it.
    """
    __slots__ = ("open_db", "close_db", "alpha")
    def __init__(self, threshold_db=-40.0, hysteresis_db=3.0, alpha=0.01):
        self.open_db = threshold_db
        self.close_db = threshold_db - hysteresis_db
        self.alpha = alpha

    def new_state(self, meta):
        return SimpleNamespace(level=None, is_open=False)

    def step(self, samples, st, meta):
        lo, hi = 10**(self.close_db/10), 10**(self.open_db/10)
        x = np.ascontiguousarray(samples)
        if _accel.native is not None and x.dtype in _NATIVE_DTYPES:
            out, lvl, have, op = _accel.native.squelch(x, 0.0 if st.level is None else float(st.level), st.level is not None,
                                                       bool(st.is_open), lo, hi, float(self.alpha))
            st.level, st.is_open = (lvl if have else None), op
            return out
        pw = np.abs(samples)**2
        out = np.zeros_like(samples)
        lvl, op = st.level, st.is_open
        for i, p in enumerate(pw):
            lvl = p if lvl is None else lvl + self.alpha*(p - lvl)
            op = lvl > (lo if op else hi)
            if op:
                out[i] = samples[i]
        st.level, st.is_open = lvl, op
        return out
