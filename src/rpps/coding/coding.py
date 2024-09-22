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
    def __init__(self, e_impl, d_impl):
        self.log = Logger().Child("Coding", Level.WARN).Child(type(self).__name__, Level.WARN)
        self._enc = e_impl
        self._dec = d_impl

    def __str__(self) -> str:
        return f"{self.name}:{self.decision.name}:{type(self._enc).__name__}:{self.num}/{self.den}"

    @property
    def num(self):
        """Return number of data bits"""
        return self._enc.num

    @property
    def den(self):
        """Return number of encoded bits"""
        return self._enc.den

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

    def encode(self, dobj):
        """Encode dobject using specified coding"""
        # print(f"Data: {self.backend.num} / total: {self.backend.den}")
        rate = (self._enc.num/self._enc.den)
        retr_len = int(len(dobj) / rate)
        itr_cnt = len(dobj)//self._enc.num
        data = np.zeros((retr_len,), dtype=bool)

        inps = self._enc.num
        oups = self._enc.den

        for i in range(0, itr_cnt, 1):
            data[i*oups : (i+1)*oups] = self._enc.encode(dobj.data[i*inps: (i+1)*inps])
        return dobject.CodingData(data)

    def decode(self, dobj):
        """Decode dobject using specified coding"""
        rate = (self._dec.num/self._dec.den)
        retr_len = int(len(dobj) * rate)
        itr_cnt = len(dobj) // self._dec.den
        data = np.zeros((retr_len,), dtype=bool)

        inps = self._dec.den
        oups = self._dec.num

        for i in range(0, itr_cnt, 1):
            data[i*oups : (i+1)*oups] = self._dec.decode(dobj.data[i*inps: (i+1)*inps])
        return dobject.BitObject(data)

    @staticmethod
    def load(name, obj):
        i_code = getattr(types, obj["type"])

        if obj["type"] == "linear":
            gen = np.array(obj["generator"], dtype=bool)
            chk = np.array(obj["check"], dtype=bool)
            i_code = i_code(gen, chk)

            impl = type(name, (Block,), dict())
            impl.name = name
            return impl(i_code, i_code)
        if obj["type"] == "repeat":
            i_code = i_code(obj["count"])

            impl = type(name, (Block,), dict())
            impl.name = name
            return impl(i_code, i_code)
        raise NotImplementedError(f"{name} is not implemented")

class Convolutional(Coding):
    """Parent convolutional coding"""

    def encode(self, dobj):
        rate = self._enc.num / self._enc.den
        retr_len = int(len(dobj) / rate)
        itr_cnt = len(dobj) // self._enc.num
        data = np.zeros((retr_len,), dtype=bool)

        inps = self._enc.num
        oups = self._enc.den

        for i in range(0, itr_cnt, 1):
            data[i*oups : (i+1)*oups] = self._enc.encode(dobj.data[i*inps: (i+1)*inps])
        return dobject.CodingData(data)

    def decode(self, dobj):
        pass

    @staticmethod
    def load(name, obj):
        if obj["type"] == "split":
            # i_d_code = getattr(types, obj["decode"]["type"])
            i_e_code = getattr(types, obj["encode"]["type"])
            gen = np.array(obj["encode"]["generator"], dtype=bool)

            i_e_code = i_e_code(1, 3, gen, obj["encode"]["constraint"])

            impl = type(name, (Convolutional,), dict())
            impl.name = name
            return impl(i_e_code, i_e_code)
        raise NotImplementedError(f"{name} is not implemented")
