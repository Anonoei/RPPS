from ..base import Meta

class Sample:
    def __radd__(self, meta: Meta):
        raise NotImplementedError()
    def __rsub__(self, meta: Meta):
        raise NotImplementedError()
