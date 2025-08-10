from .sample import Sample, Meta

class Downsample(Sample):
    __slots__ = ("filt")
    def __init__(self, ratio):
        self.ratio = ratio

class Decimate(Downsample):
    def run(self, samples):
        return samples[::int(self.ratio)]

    def __radd__(self, meta: Meta):
        meta.obj = self.run(meta.obj)
        meta.Fs /= self.ratio
        return meta
