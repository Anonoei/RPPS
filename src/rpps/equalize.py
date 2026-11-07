"""Adaptive equalizers: `symbols + rp.equalize.CMA(11)`, the taps adapt as the stream goes by

Each makes one output per symbol from `sps` input samples per symbol (1, or 2 for a fractionally spaced equalizer).
The taps carry across blocks, so any block length gives the same stream.
"""
from types import SimpleNamespace

import numpy as np

from .base import Stage

class _Equalizer(Stage):
    __slots__ = ("taps", "mu", "sps")
    def __init__(self, taps=11, mu=0.01, sps=None):
        """taps: filter length. mu: step size (smaller is steadier, larger tracks faster).
        sps: input samples per symbol (None: what the data says, rounded)"""
        if taps < 1:
            raise ValueError("taps must be at least 1")
        self.taps, self.mu, self.sps = int(taps), mu, sps

    def new_state(self, meta):
        w = np.zeros(self.taps, dtype=np.complex64)
        w[self.taps//2] = 1.0 # Start as a pass-through at the middle tap
        sps = self.sps or max(1, meta.isps)
        return SimpleNamespace(w=w, tail=np.zeros(self.taps - 1, dtype=np.complex64), next=self.taps - 1, sps=sps, n=0)

    def error(self, y, st):
        """Update direction for output y: w += mu * error * conj(x)"""
        raise NotImplementedError()

    def step(self, samples, st, meta):
        N, sps, mu = self.taps, st.sps, self.mu
        buf = np.concatenate((st.tail, samples.astype(np.complex64, copy=False)))
        out = []
        n, w = st.next, st.w
        while n < len(buf):
            x = buf[n - N + 1:n + 1][::-1]
            y = np.dot(w, x)
            w = w + mu*self.error(y, st)*np.conj(x)
            out.append(y)
            n += sps
            st.n += 1
        keep = min(n - N + 1, len(buf))
        st.tail, st.next, st.w = buf[keep:], n - keep, w
        meta.op.Fs = meta.Fs/sps
        meta.op.sps = meta.sps/sps
        return np.asarray(out, dtype=np.complex64)

class CMA(_Equalizer):
    """Constant modulus algorithm: blind, for constant-envelope modulations (PSK, FSK) -- it fixes ISI but not rotation

    modulus: the symbol amplitude it steers towards (1.0)
    """
    __slots__ = ("R",)
    def __init__(self, taps=11, mu=0.005, sps=None, modulus=1.0):
        super().__init__(taps, mu, sps)
        self.R = modulus**2

    def error(self, y, st):
        return -y*(abs(y)**2 - self.R)

class LMS(_Equalizer):
    """Least mean squares against a training sequence for the first symbols, then against the nearest point of
    `points` (decision directed). Needs a carrier that is already locked

        eq = rp.equalize.LMS(11, points=rp.mod.load("QPSK").points.arr, training=known_symbols)
    """
    __slots__ = ("points", "training")
    def __init__(self, taps=11, mu=0.01, sps=None, points=None, training=None):
        super().__init__(taps, mu, sps)
        if points is None and training is None:
            raise ValueError("LMS needs the constellation `points`, or a `training` sequence")
        self.points = None if points is None else np.asarray(points, dtype=np.complex64)
        self.training = None if training is None else np.asarray(training, dtype=np.complex64)

    def error(self, y, st):
        k = st.n
        if self.training is not None and k < len(self.training):
            d = self.training[k]
        elif self.points is not None:
            d = self.points[np.argmin(np.abs(self.points - y))]
        else:
            return 0.0
        return d - y
