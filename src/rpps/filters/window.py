import numpy as np

from .impl.window import s, get
from .filter import Filter, Meta

class Window(Filter):
    def __init__(self, name, N=None, *args, **kwargs):
        self.N = N
        self.window = get(name)
        if self.N is not None:
            self.w = self.window(self.N, *args, **kwargs)

    def __call__(self, N):
        return self.window(N)

    def run(self, N):
        return self.window(N)

    def step(self, samples, st, meta: Meta):
        """Weight each block by the window (one window per block: N, or the block's length)"""
        if self.N is None:
            return samples * self.run(len(samples))
        if len(samples) != self.N:
            raise ValueError(f"Window of {self.N} was given a block of {len(samples)}")
        return samples * self.w

    @staticmethod
    def s():
        return s

    @staticmethod
    def get(name):
        return get(name)
