import numpy as np

from ..meta import Meta

class Filter:
    def __radd__(self, meta: Meta):
        raise NotImplementedError()
    def __rsub__(self, meta: Meta):
        raise NotImplementedError()
