from .sample import Sample, Meta

# from ..filters.shaping import ShapingSinc
from ..filters import shaping

import numpy as np

class Upsample(Sample):
    __slots__ = ("ratio")
    def __init__(self, ratio):
        self.ratio = ratio

class FFT(Upsample):
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

    def __radd__(self, meta: Meta):
        meta.obj = self.run(meta.obj)
        meta.Fs *= self.ratio
        return meta

class Sinc(Upsample):
    __slots__ = ("pulse")
    def __init__(self, ratio):
        super().__init__(ratio)
        taps = np.max([self.ratio*12, 31])
        self.pulse = shaping.ShapingSinc(taps, ratio, 1) # TODO: tune this

    def run(self, samples):
        up = np.zeros(len(samples)*self.ratio, dtype=samples.dtype)
        up[::self.ratio] = samples[::1]
        up = self.pulse.run(up)
        return up

    def __radd__(self, meta: Meta):
        meta.obj = self.run(meta.obj)
        meta.Fs *= self.ratio
        return meta

class Zeros(Upsample):
    def run(self, samples):
        up = np.zeros(len(samples)*self.ratio, dtype=samples.dtype)
        up[::self.ratio] = samples[::1]
        return up

    def __radd__(self, meta: Meta):
        meta.obj = self.run(meta.obj)
        meta.Fs *= self.ratio
        return meta
