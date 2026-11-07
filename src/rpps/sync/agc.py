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

from ..base import Stage

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

AGC_log = AGC
def AGC_lin(target=1.0, mu=0.01, **kw):
    """Linear-domain AGC"""
    return AGC(target, mu, log=False, **kw)
