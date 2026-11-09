"""Analog demodulation and signal conditioning stages (FM is `rp.mod.FMDemod`)"""
from types import SimpleNamespace

import numpy as np

from . import _accel
from .base import Stage
from .filters.filter import FIR
from .filters.impl.designer import designer

_NATIVE_DTYPES = (np.float32, np.float64, np.complex64, np.complex128) # What the rpps._native loops take

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
        if _accel.native is not None:
            out, mean, have = _accel.native.am_demod(env, 0.0 if st.mean is None else float(st.mean), st.mean is not None, float(self.alpha))
            st.mean = mean if have else None
            return out
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

class DCBlock(Stage):
    """Removes the DC offset (a one pole high pass: y[n] = x[n] - x[n-1] + r*y[n-1])"""
    __slots__ = ("r",)
    def __init__(self, r=0.999):
        self.r = r

    def new_state(self, meta):
        return SimpleNamespace(x=0.0, y=0.0)

    def step(self, samples, st, meta):
        x = np.ascontiguousarray(samples)
        if _accel.native is not None and x.dtype in _NATIVE_DTYPES:
            cplx = x.dtype.kind == "c"
            out, st.x, st.y = _accel.native.dc_block(x, complex(st.x) if cplx else float(np.real(st.x)),
                                                     complex(st.y) if cplx else float(np.real(st.y)), float(self.r))
            return out
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
        x = np.ascontiguousarray(samples)
        if _accel.native is not None and x.dtype in _NATIVE_DTYPES:
            out, st.y = _accel.native.one_pole(x, complex(st.y) if x.dtype.kind == "c" else float(np.real(st.y)), float(a))
            return out
        out = np.empty_like(samples)
        y = st.y
        for i, v in enumerate(samples):
            y += a*(v - y)
            out[i] = y
        st.y = y
        return out
