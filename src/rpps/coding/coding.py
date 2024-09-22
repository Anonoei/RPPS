"""Coding parent classes"""
from abc import abstractmethod
from enum import Enum
import numpy as np

from pyboiler.logger import Logger, Level

from . import base
from . import dobject

from . import _types as types


class Decision(Enum):
    """Coding decision type"""
    HARD = 0
    SOFT = 1

class Coding(base.rpps.Pipe):
    """Coding Pipe"""
    name = "Coding"
    decision = Decision.HARD
    def __init__(self, backend):
        self.log = Logger().Child("Coding", Level.WARN).Child(type(self).__name__, Level.WARN)
        self.backend = backend

    def __str__(self) -> str:
        return f"{self.name}:{self.decision.name}:{type(self.backend).__name__}:{self.num}/{self.den}"

    @property
    def num(self):
        """Return number of data bits"""
        return self.backend.num

    @property
    def den(self):
        """Return number of encoded bits"""
        return self.backend.den

    @property
    def rate(self):
        """Return bits/parity rate"""
        return self.num/self.den

    @abstractmethod
    def encode(self, dobj: dobject.BitObject) -> dobject.CodingData:
        """Encode dobject using specified coding"""

    @abstractmethod
    def decode(self, dobj: dobject.BitObject) -> dobject.BitObject:
        """Decode dobject using specified coding"""

    @staticmethod
    @abstractmethod
    def load(name: str, obj: dict):
        """Load coding from json"""

    def __rmul__(self, other):
        return self.encode(dobject.ensure_bit(other))

    def __rtruediv__(self, other):
        if issubclass(type(other), dobject.ModData):
            if self.decision == Decision.HARD:
                other = dobject.ModData(other.hard)
        return self.decode(other)


class Block(Coding):
    """Parent block coding"""

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

    @staticmethod
    def load(name, obj):
        i_code = getattr(types, obj["type"])

        if obj["type"] == "linear":
            gen = np.array(obj["generator"], dtype=bool)
            chk = np.array(obj["check"], dtype=bool)
            impl = type(name, (Block,), dict())
            impl.name = name
            return impl(i_code(gen, chk))
        if obj["type"] == "repeat":
            impl = type(name, (Block,), dict())
            impl.name = name
            return impl(i_code(obj["count"]))
        raise NotImplementedError(f"{name} is not implemented")

class Convolutional(Coding):
    """Parent convolutional coding"""
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
