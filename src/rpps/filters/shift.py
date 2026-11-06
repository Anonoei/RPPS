from types import SimpleNamespace

import numpy as np

from ..base import Meta
from .filter import Filter

class ShiftFreq(Filter):
    """Shift the spectrum by `fo` Hz. The oscillator phase carries across blocks"""
    __slots__ = ("fo")
    def __init__(self, fo):
        self.fo = fo

    def new_state(self, meta):
        return SimpleNamespace(phase=0.0)

    def run(self, samples, Fs):
        """One-shot (stateless) shift"""
        t = np.arange(len(samples))/Fs
        return samples * np.exp(2j*np.pi*self.fo*t)

    def step(self, samples, st, meta):
        inc = 2*np.pi*self.fo/meta.Fs
        out = samples * np.exp(1j*(st.phase + inc*np.arange(len(samples))))
        st.phase = (st.phase + inc*len(samples)) % (2*np.pi)
        meta.op.cf += self.fo
        return out

class ShiftFreqIdx(ShiftFreq):
    def run(self, samples, idx):
        t = np.arange(len(samples))
        return samples * np.exp(2j*np.pi*self.fo*t)

class ShiftTime(Filter):
    def __init__(self, delay, taps=21):
        self.pulse = np.arange(-(taps-1)//2, taps//2+1)
        self.pulse = np.sinc(self.pulse - delay) # calc filter taps
        self.pulse *= np.hamming(taps)
        self.pulse /= np.sum(self.pulse)  # unity gain

    def run(self, samples):
        return np.convolve(samples, self.pulse, mode="valid")

    def __radd__(self, meta: Meta):
        meta.obj = self.run(meta.obj)
        meta.finish()
        return meta
