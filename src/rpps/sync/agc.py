"""Automatic Gain Control

R = reference level

x[n] = input signal
z[n] = a[n-1] * x[n]               controlled output
e[n] = R - |z[n]|                  (linear) or  log(R) - log|z[n]|  (log)
a[n] = a[n-1] + mu * e[n]          (linear) or  a[n-1] * exp(mu * e[n])  (log)

The gain is carried across blocks, so any block length gives the same stream.
"""
from types import SimpleNamespace

import numpy as np

from .. import _accel
from ..base import Stage

class PeakAGC(Stage):
    """Gain from the peak: |x| is followed instantly upward and decays by `decay` per sample, and the signal is scaled so
    that peak sits at `target`. For bursts and on-off keying (Mode S pulses, OOK): the first pulse of a burst sets the level,
    where a loop that tracks the mean power is still settling. decay 0.999 at 2 MS/s holds a peak for about a millisecond"""
    __slots__ = ("target", "decay", "floor")
    def __init__(self, target=1.0, decay=0.999, floor=1e-6):
        self.target, self.decay, self.floor = float(target), float(decay), float(floor)

    def new_state(self, meta):
        return SimpleNamespace(level=0.0)

    def step(self, samples, st, meta):
        x = np.asarray(samples)
        mag = np.abs(x).astype(np.float64)
        level, d = st.level, self.decay
        out = np.empty(len(x), dtype=np.float64)
        for i, m in enumerate(mag): # (a short loop: one max and one multiply per sample)
            level = m if m > level*d else level*d
            out[i] = level
        st.level = level
        return (x*(self.target/np.maximum(out, self.floor))).astype(x.dtype)


class AGC(Stage):
    """Feedback AGC: `block + AGC(1.0, 0.01)`

    target: output amplitude R
    mu: loop rate (per sample). Larger tracks faster and ripples more
    log: adapt the gain in the log domain, so the loop speed doesn't depend on the signal level
    max_gain/min_gain: limits on the gain
    """
    __slots__ = ("target", "mu", "log", "max", "min", "init")
    def __init__(self, target=1.0, mu=0.01, log=True, min_gain=0.0, max_gain=np.inf, init=1.0):
        self.target = target
        self.mu = mu
        self.log = log
        self.min = min_gain
        self.max = max_gain
        self.init = init

    def new_state(self, meta):
        return SimpleNamespace(gain=self.init)

    def step(self, samples, st, meta):
        if _accel.native is not None:
            x = np.ascontiguousarray(samples)
            if x.dtype not in (np.complex64, np.complex128, np.float32, np.float64):
                x = x.astype(np.float32)
            out, st.gain = _accel.native.agc(x, float(st.gain), float(self.mu), float(self.target), float(self.min), float(self.max), bool(self.log))
            return out
        out = np.empty(len(samples), dtype=samples.dtype if samples.dtype.kind == "c" else np.float32)
        gain, mu, target, lo, hi = st.gain, self.mu, self.target, self.min, self.max
        amp = np.abs(samples)
        if self.log:
            ltarget = np.log(target)
            for i, a in enumerate(amp):
                z = gain*a
                out[i] = gain*samples[i]
                gain = min(max(gain*np.exp(mu*(ltarget - np.log(z + 1e-30))), lo), hi)
        else:
            for i, a in enumerate(amp):
                out[i] = gain*samples[i]
                gain = min(max(gain + mu*(target - gain*a), lo), hi)
        st.gain = gain
        return out


