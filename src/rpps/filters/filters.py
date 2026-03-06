import numpy as np

from .impl.designer import designer
from .filter import Filter, Meta

class ConvFilter(Filter):
    __slots__ = ("filt")
    def __init__(self, filt):
        self.filt = filt

    def run(self, samples):
        return np.convolve(samples, self.filt, mode="valid")

    def __radd__(self, meta: Meta):
        meta.obj = self.run(meta.obj)
        meta.finish()
        return meta

def low_pass(taps, cutoff, fs=None):
    return ConvFilter(designer(taps, cutoff, fs=fs, pass_zero=True))

def high_pass(taps, cutoff, fs=None):
    return ConvFilter(designer(taps, cutoff, fs=fs, pass_zero=False))

def band_pass(taps, cut1, cut2, fs=None):
    return ConvFilter(designer(taps, [cut1, cut2], fs=fs, pass_zero=False))

def band_stop(taps, cut1, cut2, fs=None):
    return ConvFilter(designer(taps, [cut1, cut2], fs=fs, pass_zero=False))
