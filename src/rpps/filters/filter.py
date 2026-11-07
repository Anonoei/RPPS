from types import SimpleNamespace

import numpy as np

from ..base import Meta, Stage

class Filter(Stage):
    pass

class FIR(Filter):
    """Causal FIR that carries its history across blocks: len(out) == len(in)"""
    def make_taps(self, meta: Meta):
        raise NotImplementedError()

    def new_state(self, meta: Meta):
        return SimpleNamespace(taps=np.asarray(self.make_taps(meta)), hist=None)

    @staticmethod
    def stream(samples, taps, hist):
        """Filter a block. Returns (output, new history)"""
        keep = len(taps) - 1
        if hist is None:
            hist = np.zeros(keep, dtype=samples.dtype)
        buf = np.concatenate((hist, samples))
        if len(samples) == 0:
            return samples, hist
        out = np.convolve(buf, taps, mode="valid")
        return out, buf[len(buf)-keep:]

    def step(self, samples, st, meta: Meta):
        out, st.hist = self.stream(samples, st.taps, st.hist)
        meta.delay += (len(st.taps)-1) / 2 / meta.Fs
        return out
