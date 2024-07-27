from pyboiler.logger import Logger, Level

from ..helpers.bitarray import bitarray
from ..helpers.stream import Stream

from . import Meta

class Coding:
    def __init__(self, num, den):
        self.log = Logger().Child("Coding", Level.WARN).Child(type(self).__name__, Level.TRACE)
        self.num = num
        self.den = den
        self._rate = self.num/self.den

    def init_meta(self, meta: Meta):
        meta.coding.fields["Name"] = type(self).__name__
        meta.coding.fields["RateNum"] = self.num
        meta.coding.fields["RateDen"] = self.den

    def encode(self, data: Stream) -> bitarray:
        ...

    def decode(self, data: bitarray) -> bitarray:
        ...


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
