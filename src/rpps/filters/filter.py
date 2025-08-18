import numpy as np

from ..base import Meta, Pipe

class Filter(Pipe):
    def __radd__(self, meta: Meta):
        raise NotImplementedError()
    def __rsub__(self, meta: Meta):
        raise NotImplementedError()
