import numpy as np

from .impl.designer import designer
from .filter import FIR, Meta

class ConvFilter(FIR):
    __slots__ = ("filt")
    def __init__(self, filt):
        self.filt = filt

    def make_taps(self, meta):
        return self.filt

    def run(self, samples):
        """One-shot (stateless) filtering, drops the edges"""
        return np.convolve(samples, self.filt, mode="valid")

def low_pass(taps, cutoff, fs=None):
    return ConvFilter(designer(taps, cutoff, fs=fs, pass_zero=True))

def high_pass(taps, cutoff, fs=None):
    return ConvFilter(designer(taps, cutoff, fs=fs, pass_zero=False))

def band_pass(taps, cut1, cut2, fs=None):
    return ConvFilter(designer(taps, [cut1, cut2], fs=fs, pass_zero=False))

def band_stop(taps, cut1, cut2, fs=None):
    return ConvFilter(designer(taps, [cut1, cut2], fs=fs, pass_zero=False))
