"""Squelch: detects that a signal is present, and gates the stream on it"""
from types import SimpleNamespace

import numpy as np

from ..base import Stage
from ..signal.filters.iir import RunningMean

class Squelch(Stage):
    """Silence the signal while it is quieter than `threshold_db` (power, dB re 1.0), with hysteresis

    The level follows a running average of the power (`alpha` of the gap a sample, from the first sample on: an IIR
    section), so a brief dip doesn't close it. It opens where the level is over the open threshold, closes where it is at
    or under the close one, and in between stays as it was: the last of those events, carried forward (no loop over the
    samples).
    """
    __slots__ = ("open_db", "close_db", "alpha")
    def __init__(self, threshold_db=-40.0, hysteresis_db=3.0, alpha=0.01):
        self.open_db = threshold_db
        self.close_db = threshold_db - hysteresis_db
        self.alpha = alpha

    def new_state(self, meta):
        return SimpleNamespace(mean=RunningMean(self.alpha), z=None, is_open=False)

    def step(self, samples, st, meta):
        lo, hi = 10**(self.close_db/10), 10**(self.open_db/10)
        x = np.asarray(samples)
        level, st.z = st.mean.run(np.abs(x).astype(np.float64)**2, st.z)
        is_open = gate(level, lo, hi, st.is_open)
        if len(is_open):
            st.is_open = bool(is_open[-1])
        return np.where(is_open, x, np.zeros((), x.dtype))

def gate(level, lo, hi, was_open=False):
    """Hysteresis: open where level > hi, closed where level <= lo, else as at the sample before (`was_open` before the
    first): each sample takes the last decisive one's state"""
    level = np.asarray(level)
    up, down = level > hi, level <= lo
    decisive = up | down
    last = np.maximum.accumulate(np.where(decisive, np.arange(len(level)), -1)) if len(level) else np.zeros(0, int)
    return np.where(last >= 0, up[np.maximum(last, 0)], bool(was_open))
