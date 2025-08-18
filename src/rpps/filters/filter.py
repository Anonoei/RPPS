import numpy as np

from ..base.rpps import Pipe
from ..meta import Meta

class Filter(Pipe):
    def __radd__(self, meta: Meta):
        raise NotImplementedError()
    def __rsub__(self, meta: Meta):
        raise NotImplementedError()
