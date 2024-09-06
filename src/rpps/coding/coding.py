from pyboiler.logger import Logger, Level

from . import Meta
from . import base
from . import dobject

class Coding:
    name = "Coding"
    def __init__(self, num, den):
        self.log = Logger().Child("Coding", Level.WARN).Child(type(self).__name__, Level.WARN)
        self.num = num
        self.den = den
        self._rate = self.num/self.den

    def init_meta(self, meta: Meta):
        meta.coding.fields["Name"] = type(self).__name__
        meta.coding.fields["RateNum"] = self.num
        meta.coding.fields["RateDen"] = self.den

    def encode(self, dobj: dobject.DataObject, meta: Meta) -> dobject.CodingData:
        ...

    def decode(self, dobj: dobject.DataObject, meta: Meta) -> dobject.CodingData:
        ...

    def __matmul__(self, other):
        if isinstance(other, dobject.ModData):
            return self.decode(other, other.meta)
        elif issubclass(type(other), dobject.DataObject):
            return self.encode(other, other.meta)
        else:
            raise TypeError(f"Cannot perform {type(self).__name__} on {type(other)}")

    def __rmatmul__(self, other):
        if isinstance(other, dobject.ModData):
            return self.decode(other, other.meta)
        elif issubclass(type(other), dobject.DataObject):
            return self.encode(other, other.meta)
        else:
            raise TypeError(f"Cannot perform {type(self).__name__} on {type(other)}")


class Block(Coding):
    def __init__(self, num, den, length):
        super().__init__(num, den)
        self.length = length

    def init_meta(self, meta: Meta):
        from .meta import BlockCodingMeta
        meta.coding = BlockCodingMeta()  # type: ignore
        meta.coding.fields["Type"] = "BLK"
        meta.coding.fields["Length"] = self.length
        super().init_meta(meta)


class Convolutional(Coding):

    def init_meta(self, meta: Meta):
        from .meta import ConvolutionalCodingMeta
        meta.coding = ConvolutionalCodingMeta()  # type: ignore
        meta.coding.fields["Type"] = "CNV"
        super().init_meta(meta)
