"""Coding parent classes"""
from pyboiler.logger import Logger, Level

from . import Meta
from . import base
from . import dobject

from enum import Enum

class Decision(Enum):
    """Coding decision type"""
    HARD = 0
    SOFT = 1

class Coding(base.rpps.Pipe):
    """Coding Pipe"""
    name = "Coding"
    decision = Decision.HARD
    def __init__(self, num, den):
        self.log = Logger().Child("Coding", Level.WARN).Child(type(self).__name__, Level.WARN)
        self.num = num
        self.den = den
        self._rate = self.num/self.den

    def __str__(self):
        return f"{self.name}:{self.num}/{self.den}:{self.decision}"

    def init_meta(self, meta: Meta):
        """Initialize coding metadata"""
        meta.coding.fields["Name"] = type(self).__name__
        meta.coding.fields["RateNum"] = self.num
        meta.coding.fields["RateDen"] = self.den

    def encode(self, dobj: dobject.BitObject) -> dobject.CodingData:
        """Encode dobject using specified coding"""
        ...

    def decode(self, dobj: dobject.BitObject) -> dobject.BitObject:
        """Decode dobject using specified coding"""
        ...

    def __matmul__(self, other):
        if isinstance(other, dobject.ModData):
            if self.decision == Decision.HARD:
                other = dobject.ModData(other.hard)
            return self.decode(other)
        elif issubclass(type(other), dobject.DataObject):
            return self.encode(dobject.ensure_bit(other))
        else:
            raise TypeError(f"Cannot perform {type(self).__name__} on {type(other)}")

    def __rmatmul__(self, other):
        return self.__matmul__(other)


class Block(Coding):
    """Parent block coding"""
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
    """Parent convolutional coding"""
    def init_meta(self, meta: Meta):
        from .meta import ConvolutionalCodingMeta
        meta.coding = ConvolutionalCodingMeta()  # type: ignore
        meta.coding.fields["Type"] = "CNV"
        super().init_meta(meta)
