"""Coding parent classes"""
from pyboiler.logger import Logger, Level

from . import Meta
from . import base
from . import dobject

from . import _types as types

from enum import Enum
import numpy as np

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

    def __add__(self, other):
        return self.encode(dobject.ensure_bit(other))

    def __sub__(self, other):
        if issubclass(type(other), dobject.ModData):
            if self.decision == Decision.HARD:
                other = dobject.ModData(other.hard)
        return self.decode(other)


class Block(Coding):
    """Parent block coding"""
    def __init__(self, backend):
        self.backend = backend
        self.length = self.backend.den
        super().__init__(self.backend.num, self.backend.den)

    def encode(self, dobj: dobject.BitObject) -> dobject.CodingData:
        """Encode dobject using specified coding"""
        # print(f"Data: {self.backend.num} / total: {self.backend.den}")
        rate = (self.backend.num/self.backend.den)
        retr_len = int(len(dobj) / rate)
        itr_cnt = len(dobj)//self.backend.num
        data = np.zeros((retr_len,), dtype=bool)

        inps = self.backend.num
        oups = self.backend.den

        for i in range(0, itr_cnt, 1):
            data[i*oups : (i+1)*oups] = self.backend.encode(dobj.data[i*inps: (i+1)*inps])
        return dobject.CodingData(data)

    def decode(self, dobj: dobject.BitObject) -> dobject.BitObject:
        """Decode dobject using specified coding"""
        rate = (self.backend.num/self.backend.den)
        retr_len = int(len(dobj) * rate)
        itr_cnt = len(dobj) // self.backend.den
        data = np.zeros((retr_len,), dtype=bool)

        inps = self.backend.den
        oups = self.backend.num

        for i in range(0, itr_cnt, 1):
            data[i*oups : (i+1)*oups] = self.backend.decode(dobj.data[i*inps: (i+1)*inps])
        return dobject.BitObject(data)

    def init_meta(self, meta: Meta):
        from .meta import BlockCodingMeta
        meta.coding = BlockCodingMeta()  # type: ignore
        meta.coding.fields["Type"] = "BLK"
        meta.coding.fields["Length"] = self.length
        super().init_meta(meta)

    @staticmethod
    def load(name, obj):
        i_code = getattr(types, obj["type"])

        if obj["type"] == "linear":
            gen = np.array(obj["generator"], dtype=bool)
            chk = np.array(obj["check"], dtype=bool)
            return type(name, (Block,), dict())(i_code(gen, chk))
        raise NotImplementedError(f"{name} is not implemented")

class Convolutional(Coding):
    """Parent convolutional coding"""
    def init_meta(self, meta: Meta):
        from .meta import ConvolutionalCodingMeta
        meta.coding = ConvolutionalCodingMeta()  # type: ignore
        meta.coding.fields["Type"] = "CNV"
        super().init_meta(meta)
    def __init__(self, k, passthrough, polys):
        self.input_size = k
        assert len(passthrough) == polys.shape[0]
        self.passthrough = passthrough
        self.polys = polys
        self.output_size = polys.shape[0]
        self.register = np.zeros((self.output_size), dtype=bool)

    def code(self, bits):
        assert len(bits) == self.input_size

        output = np.zeros(self.output_size, dtype=bool)
        for i, p in enumerate(self.polys):
            if self.passthrough[i]:
                output[i] = bits[i]
                continue
            output[i] = np.bitwise_xor.reduce(self.register[p])

        self.register[0] = bits[0]
        self.register = np.roll(self.register, 1)
        return output

    @staticmethod
    def load(name, obj):
        i_code = getattr(types, obj["type"])

        if obj["type"] == "linear":
            gen = np.array(obj["generator"], dtype=bool)
            chk = np.array(obj["check"], dtype=bool)
            return type(name, (Block,), dict())(i_code(gen, chk))
        raise NotImplementedError(f"{name} is not implemented")
