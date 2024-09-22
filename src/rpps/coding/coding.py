"""Coding parent classes"""
from abc import abstractmethod
from enum import Enum
import numpy as np

from pyboiler.logger import Logger, Level

from . import base
from . import dobject
from .blocker import unblock

from . import types as types


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
        return dobject.CodingData(self._enc.encode(dobj.data))

    def decode(self, dobj):
        """Decode dobject using specified coding"""
        return dobject.BitObject(self._dec.decode(dobj.data))

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
        return dobject.CodingData(self._enc.encode(dobj.data))

    def decode(self, dobj):
        return dobject.BitObject(self._dec.decode(dobj.data))

    @staticmethod
    def load(name, obj):
        if obj["type"] == "split":
            i_e_code = getattr(types, obj["encode"]["type"])
            i_d_code = getattr(types, obj["decode"]["type"])

            i_e_num = obj["encode"]["num"]
            i_e_den = obj["encode"]["den"]
            i_e_gen = np.array(obj["encode"]["generator"], dtype=bool)
            i_e_con = obj["encode"]["constraint"]
            i_e_code = i_e_code(i_e_num, i_e_den, i_e_gen, i_e_con)

            i_d_num = obj["decode"]["num"]
            i_d_den = obj["decode"]["den"]
            i_d_con = obj["decode"]["constraint"]
            i_d_code = i_d_code(i_d_num, i_d_den, i_d_con)

            impl = type(name, (Convolutional,), dict())
            impl.name = name
            return impl(i_e_code, i_d_code)
        raise NotImplementedError(f"{name} is not implemented")
