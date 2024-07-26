from pyboiler.logger import Logger, Level

from ..helpers.bitarray import bitarray
from ..helpers.stream import Stream

class Coding:
    def __init__(self, num, den):
        self.log = Logger().Child("Coding", Level.WARN).Child(type(self).__name__, Level.TRACE)
        self.num = num
        self.den = den
        self._rate = self.num/self.den

    def encode(self, data: Stream) -> bitarray:
        ...

    def decode(self, data: bitarray) -> bitarray:
        ...
