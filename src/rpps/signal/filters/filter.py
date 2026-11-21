from types import SimpleNamespace

import numpy as np

from ...base import Meta, Stage
from ... import _accel
from .convolve import convolve, fft_pays

class Filter(Stage):
    pass

NATIVE_TAPS = 256 # Up to this many taps the native direct filter is faster than FFT blocks at any block length

class FIR(Filter):
    """Causal FIR that carries its history across blocks: len(out) == len(in)"""
    def make_taps(self, meta: Meta):
        raise NotImplementedError()

    def new_state(self, meta: Meta):
        taps = np.asarray(self.make_taps(meta))
        return SimpleNamespace(taps=taps, hist=None)

    @staticmethod
    def stream(samples, taps, hist):
        """Filter a block. Returns (output, new history)"""
        keep = len(taps) - 1
        if hist is None:
            hist = np.zeros(keep, dtype=samples.dtype)
        if len(samples) == 0:
            return samples, hist
        buf = np.concatenate((hist, samples))
        taps = np.asarray(taps)
        if (_accel.native is not None and buf.dtype == np.complex64 and taps.dtype.kind == "f"
                and (len(taps) <= NATIVE_TAPS or not fft_pays(len(buf), len(taps)))):
            # The C++ dot products (vectorised, on several threads) beat FFT blocks up to a few hundred taps, and
            # np.convolve (neither) wherever that would run
            out = _accel.native.fir_decimate(buf, np.ascontiguousarray(taps, dtype=np.float64), keep, 1, len(samples))
        else:
            out = convolve(buf, taps, "valid") # FFT blocks for a long filter on a long block, else np.convolve
        return out, buf[len(buf)-keep:]

    def step(self, samples, st, meta: Meta):
        out, st.hist = self.stream(samples, st.taps, st.hist)
        meta.delay += (len(st.taps)-1) / 2 / meta.Fs
        return out
