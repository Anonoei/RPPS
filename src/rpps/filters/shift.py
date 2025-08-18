import numpy as np

from ..base import Meta
from .filter import Filter

class ShiftFreq(Filter):
    __slots__ = ("fo")
    def __init__(self, fo):
        self.fo = fo

    def run(self, samples, Fs):
        t = np.arange(len(samples))/Fs
        return samples * np.exp(2j*np.pi*self.fo*t)

    def __radd__(self, meta: Meta):
        meta.obj = self.run(meta.obj, meta.Fs)
        return meta
