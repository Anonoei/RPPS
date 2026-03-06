import numpy as np

from .sample import Sample, Meta
from ..filters import low_pass

class Resample(Sample):
    __slots__ = (
        "up", "down",
        "low_pass"
    )
    def __init__(self, up, down):
        self.up = int(up)
        self.down = int(down)
        min_rate = min(self.up, self.down)
        cutoff = 1/min_rate
        self.low_pass = low_pass(101, cutoff)

    def run(self, samples):
        raise NotImplementedError()

    def __radd__(self, meta: Meta):
        meta.obj = self.run(meta.obj)
        meta.op.Fs *= self.up/self.down
        meta.finish()
        return meta

class Polyphase(Resample):
    def run(self, samples):
        out = np.zeros(len(samples)*self.up, dtype=samples.dtype)
        out[::1*self.up] = samples[::1]
        out = self.low_pass.run(out)
        out = out[::1*self.down]
        return out
