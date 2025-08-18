from ..base import Meta

class Sync:
    __slots__ = ("impl")
    def __init__(self, impl):
        self.impl = impl
    def __radd__(self, meta: Meta):
        meta = self.impl.burst(meta)
        return meta

    def __rsub__(self, meta: Meta):
        meta = self.impl.burst(meta)
        return meta
