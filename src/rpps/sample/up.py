from types import SimpleNamespace

from .sample import Sample, Meta

# from ..filters.shaping import ShapingSinc
from ..filters import shaping

import numpy as np

class Upsample(Sample):
    __slots__ = ("ratio")
    def __init__(self, ratio):
        self.ratio = int(ratio)

    def run(self, samples):
        raise NotImplementedError()

    def step(self, samples, st, meta: Meta):
        meta.op.Fs *= self.ratio
        meta.op.sps *= self.ratio
        return self.run(samples)

class FFT(Upsample):
    """Upsample each block through the FFT (the block is treated as one period, so edges wrap)"""
    def run(self, samples):
        frq = np.fft.fft(samples)
        up = self.ratio*np.hstack([
            frq[:len(frq)//2],
            frq[len(frq)//2]/self.ratio,
            np.zeros((len(frq)*self.ratio)-len(frq)-1),
            frq[len(frq)//2]/self.ratio,
            frq[-(len(frq)//2-1):]
        ])

        up = np.fft.ifft(up)
        return up

class Sinc(Upsample):
    """Zero stuff and interpolate with a windowed sinc. The filter history carries across blocks"""
    __slots__ = ("pulse", "taps")
    def __init__(self, ratio, taps=None):
        super().__init__(ratio)
        taps = max(self.ratio*12, 31) if taps is None else int(taps)
        self.pulse = shaping.ShapingSinc(taps, ratio, 1) # One-shot kernel
        self.taps = np.sinc((np.arange(taps | 1) - (taps | 1)//2)/ratio) # Streaming kernel: symmetric, centre tap

    def new_state(self, meta):
        return SimpleNamespace(hist=None)

    def run(self, samples):
        """One-shot (stateless) interpolation, same length as ratio*samples"""
        up = np.zeros(len(samples)*self.ratio, dtype=samples.dtype)
        up[::self.ratio] = samples
        up = self.pulse.run(up)
        return up

    def step(self, samples, st, meta: Meta):
        up = np.zeros(len(samples)*self.ratio, dtype=samples.dtype)
        up[::self.ratio] = samples
        out, st.hist = shaping.FIR.stream(up, self.taps, st.hist)
        meta.delay += (len(self.taps)-1) / 2 / (meta.Fs*self.ratio)
        meta.op.Fs *= self.ratio
        meta.op.sps *= self.ratio
        return out

class Zeros(Upsample):
    def run(self, samples):
        up = np.zeros(len(samples)*self.ratio, dtype=samples.dtype)
        up[::self.ratio] = samples[::1]
        return up
