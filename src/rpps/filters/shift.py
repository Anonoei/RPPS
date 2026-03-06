import numpy as np

from ..base import Meta
from .filter import Filter

class ShiftFreq(Filter):
    __slots__ = ("fo")
    def __init__(self, fo):
        self.fo = fo

    def run(self, samples, Fs):
        N = len(samples)
        t = np.arange(N)/-Fs
        return samples * np.exp(2j*np.pi*self.fo*t)

    def __radd__(self, meta: Meta):
        meta.obj = self.run(meta.obj, meta.Fs)
        meta.op.cf += self.fo
        meta.finish()
        return meta

class ShiftFreqIdx(ShiftFreq):
    def run(self, samples, idx):
        t = np.arange(len(samples))
        return samples * np.exp(2j*np.pi*self.fo*t)
