from types import SimpleNamespace

import numpy as np

from ...base import Meta, Stage
from .convolve import convolve

class Filter(Stage):
    pass

class FIR(Filter):
    """Causal FIR that carries its history across blocks: len(out) == len(in)"""
    def make_taps(self, meta: Meta):
        raise NotImplementedError()

    matched = False # A matched filter (a profile sets it): its taps are scaled so that with the pulse it matches, the symbols come back at unit gain

    def new_state(self, meta: Meta):
        taps = np.asarray(self.make_taps(meta))
        if self.matched:
            taps = taps/np.sum(taps*taps) # pulse * matched filter peaks at sum(h^2): taken off, so a symbol of 1 comes back as 1
        return SimpleNamespace(taps=taps, hist=None)

    @staticmethod
    def stream(samples, taps, hist):
        """Filter a block. Returns (output, new history)"""
        keep = len(taps) - 1
        if hist is None:
            hist = np.zeros(keep, dtype=samples.dtype)
        buf = np.concatenate((hist, samples))
        if len(samples) == 0:
            return samples, hist
        out = convolve(buf, taps, "valid") # FFT blocks for a long filter on a long block, else np.convolve
        return out, buf[len(buf)-keep:]

    def step(self, samples, st, meta: Meta):
        out, st.hist = self.stream(samples, st.taps, st.hist)
        meta.delay += (len(st.taps)-1) / 2 / meta.Fs
        return out
