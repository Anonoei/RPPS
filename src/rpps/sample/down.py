from types import SimpleNamespace

import numpy as np

from .sample import Sample, Meta

class Downsample(Sample):
    __slots__ = ("ratio")

class Decimate(Downsample):
    """Keep every `ratio`th sample. The phase carries across blocks, so any block length works"""
    def __init__(self, ratio: int):
        self.ratio = int(ratio)

    def new_state(self, meta):
        return SimpleNamespace(off=0) # Index in the next block of the next sample to keep

    def run(self, samples):
        """One-shot (stateless) decimation"""
        return samples[::self.ratio]

    def step(self, samples, st, meta):
        kept = len(range(st.off, len(samples), self.ratio))
        out = samples[st.off::self.ratio]
        st.off = st.off + kept*self.ratio - len(samples)
        meta.op.Fs /= self.ratio
        meta.op.sps /= self.ratio
        return out

class FractionalSpline(Downsample):
    def __init__(self, ratio: float):
        if ratio > 1.0:
            raise Exception("Downsample ratio must be below 1.0")
        self.ratio = ratio

    def run(self, samples):
        N = len(samples)
        int_part = int(N * self.ratio)
        # frac_part = self.ratio - int_part / N

        out = np.zeros(int_part + 1, dtype=samples.dtype)

        for i in range(int_part):
            pos = i * self.ratio
            val = np.interp(pos, [i, i + 1], samples[i:i+2],
                            left=samples[i], right=samples[i+1],
                            period=1)
            out[i] = val

        return out

    def __radd__(self, meta: Meta):
        meta.obj = self.run(meta.obj)
        meta.op.Fs *= self.ratio
        meta.op.sps *= self.ratio
        meta.finish()
        return meta
