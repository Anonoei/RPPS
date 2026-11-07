"""Analog demodulation and signal conditioning stages (FM is `rp.mod.FMDemod`)"""
from types import SimpleNamespace

import numpy as np

from .base import Stage
from .filters.filter import FIR
from .filters.impl.designer import designer

class AMDemod(Stage):
    """Envelope detector: |x| with the carrier (DC) removed by a slow running mean"""
    __slots__ = ("alpha",)
    def __init__(self, alpha=1e-3):
        """alpha: how fast the mean follows (fraction of the gap closed per sample)"""
        self.alpha = alpha

    def new_state(self, meta):
        return SimpleNamespace(mean=None)

    def step(self, samples, st, meta):
        env = np.abs(samples).astype(np.float32)
        out = np.empty_like(env)
        mean = st.mean
        for i, v in enumerate(env):
            mean = v if mean is None else mean + self.alpha*(v - mean)
            out[i] = v - mean
        st.mean = mean
        return out

class SSBDemod(FIR):
    """Single sideband: keeps `bw` Hz of the upper (or lower) side of the carrier at 0 Hz and returns the audio (real)

        audio = baseband + rp.demod.SSBDemod(2800, "usb")
    """
    __slots__ = ("bw", "sideband", "n")
    def __init__(self, bw=2800.0, sideband="usb", taps=201):
        if sideband not in ("usb", "lsb"):
            raise ValueError("sideband must be usb or lsb")
        self.bw, self.sideband, self.n = bw, sideband, taps | 1

    def make_taps(self, meta):
        lp = designer(self.n, self.bw/2, fs=meta.Fs).astype(np.float64)
        k = np.arange(self.n) - (self.n - 1)/2
        sign = 1 if self.sideband == "usb" else -1
        return (lp*np.exp(2j*np.pi*sign*(self.bw/2)*k/meta.Fs)).astype(np.complex64)

    def step(self, samples, st, meta):
        return np.real(super().step(samples, st, meta)).astype(np.float32)

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
        pw = np.abs(samples)**2
        out = np.zeros_like(samples)
        lvl, op = st.level, st.is_open
        lo, hi = 10**(self.close_db/10), 10**(self.open_db/10)
        for i, p in enumerate(pw):
            lvl = p if lvl is None else lvl + self.alpha*(p - lvl)
            op = lvl > (lo if op else hi)
            if op:
                out[i] = samples[i]
        st.level, st.is_open = lvl, op
        return out

class DCBlock(Stage):
    """Removes the DC offset (a one pole high pass: y[n] = x[n] - x[n-1] + r*y[n-1])"""
    __slots__ = ("r",)
    def __init__(self, r=0.999):
        self.r = r

    def new_state(self, meta):
        return SimpleNamespace(x=0.0, y=0.0)

    def step(self, samples, st, meta):
        out = np.empty_like(samples)
        x1, y1, r = st.x, st.y, self.r
        for i, v in enumerate(samples):
            y1 = v - x1 + r*y1
            x1 = v
            out[i] = y1
        st.x, st.y = x1, y1
        return out

class Deemphasis(Stage):
    """FM broadcast de-emphasis: a one pole low pass with time constant tau (75 us in the Americas, 50 us elsewhere)"""
    __slots__ = ("tau",)
    def __init__(self, tau=75e-6):
        self.tau = tau

    def new_state(self, meta):
        return SimpleNamespace(y=0.0)

    def step(self, samples, st, meta):
        a = 1 - np.exp(-1/(self.tau*meta.Fs))
        out = np.empty_like(samples)
        y = st.y
        for i, v in enumerate(samples):
            y += a*(v - y)
            out[i] = y
        st.y = y
        return out
