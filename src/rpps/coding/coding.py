"""Coding parent classes"""
from abc import abstractmethod
from enum import Enum
import numpy as np

from pyboiler.logger import Logger, Level

from . import base
from ..base import Analog, Digital, ptype, dtype
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

    def encode(self, data):
        """Encode data using specified coding"""
        data.as_bits()
        data.data = self._enc.encode(data.data)
        data.PT = ptype.CODED
        data.DT = dtype.bits
        return data

    def decode(self, data):
        """Decode data using specified coding"""
        data.data = self._dec.decode(data.data)
        data.PT = ptype.CODED
        data.DT = dtype.bits
        return data

    @staticmethod
    @abstractmethod
    def load(name: str, obj: dict):
        """Load coding from json"""

    def __rmul__(self, data):
        return self.encode(data)

    def __rtruediv__(self, data):
        if isinstance(data, Analog):
            if self.decision == Decision.HARD: # TODO: fix this
                data = Analog(data, ptype.MAPPED_HARD)
        return self.decode(data)

class Block(Coding):
    """Parent block coding"""

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
            i_d_gen = np.array(obj["decode"]["generator"], dtype=bool)
            i_d_code = i_d_code(i_d_num, i_d_den, i_d_con, i_d_gen)

            impl = type(name, (Convolutional,), dict())
            impl.name = name
            return impl(i_e_code, i_d_code)
        raise NotImplementedError(f"{name} is not implemented")
