"""Analog demodulation, carrier modulation to baseband: AM envelope and SSB (FM is `FMDemod`)"""
from types import SimpleNamespace

import numpy as np

from .. import _accel
from ..base import Stage
from ..signal.filters.filter import FIR
from ..signal.filters.impl.designer import designer

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

        audio = baseband + rp.wave.SSBDemod(2800, "usb")
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
