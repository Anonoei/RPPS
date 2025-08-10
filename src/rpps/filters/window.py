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

    def __radd__(self, meta: Meta):
        if self.N is None:
            meta.obj = meta.obj * self.run(len(meta.obj))
        else:
            meta.obj = meta.obj * self.w
        return meta

    @staticmethod
    def s():
        return s

    @staticmethod
    def get(name):
        return get(name)
