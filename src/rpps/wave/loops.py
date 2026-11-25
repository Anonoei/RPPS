"""Demodulation as per-sample loops (second class): `AMDemod` with its running mean run sample by sample, in the C++
kernel when it is built (rpps._native.am_demod) and a Python loop otherwise

The first-class rpps.wave.AMDemod runs the same mean as an IIR section with no loop over the samples; this gives the
same output to rounding.
"""
from types import SimpleNamespace

import numpy as np

from .. import _accel
from ..base import Stage

class AMDemod(Stage):
    """Envelope detector: |x| with the carrier (DC) removed by a slow running mean (alpha: how fast the mean follows, the
    fraction of the gap closed per sample), updated sample by sample; `AMDemod(alpha=None)` keeps the envelope"""
    __slots__ = ("alpha",)
    def __init__(self, alpha=1e-3):
        self.alpha = alpha

    def new_state(self, meta):
        return SimpleNamespace(mean=None)

    def step(self, samples, st, meta):
        env = np.abs(samples).astype(np.float32)
        if self.alpha is None:
            return env
        if _accel.native is not None:
            out, mean, have = _accel.native.am_demod(env, 0.0 if st.mean is None else float(st.mean), st.mean is not None, float(self.alpha))
            st.mean = mean if have else None
            return out
        out = np.empty_like(env)
        mean = st.mean
        for i, v in enumerate(env.astype(np.float64)): # (in double, as the kernel)
            mean = v if mean is None else mean + self.alpha*(v - mean)
            out[i] = v - mean
        st.mean = mean
        return out
