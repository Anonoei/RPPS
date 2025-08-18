"""Coding parent classes"""
from abc import abstractmethod
from enum import Enum
import numpy as np

from pyboiler.logger import Logger, Level

from . import base
from .. import _config
from ..base import Analog, Mod, Digital, pproc, dtype
from .blocker import unblock

from . import types as types


class Decision(Enum):
    """Coding decision type"""
    HARD = 0
    SOFT = 1

class Coding(base.rpps.Pipe):
    """Coding Pipe"""
    __slots__ = ("log", "_enc", "_dec")
    name = "Coding"
    decision = Decision.HARD
    def __init__(self, e_impl, d_impl):
        self.log = Logger().Child("Coding", _config.LOG_CODING).Child(type(self).__name__, _config.LOG_CODING)
        self._enc = e_impl
        self._dec = d_impl

    def __str__(self) -> str:
        return f"{self.name}:{self.decision.name}:{type(self._enc).__name__}_{self.e_num}/{self.e_den}:{type(self._dec).__name__}_{self.d_num}/{self.d_den}"

    @property
    def e_num(self):
        """Return number of encoding data bits"""
        return self._enc.num
    @property
    def e_den(self):
        """Return number of total encoded bits"""
        return self._enc.den

    @property
    def d_num(self):
        """Return number of decoding data bits"""
        return self._dec.num

    @property
    def d_den(self):
        """Return number of total decoded bits"""
        return self._dec.den

    @property
    def e_rate(self):
        """Return encoding bits/parity rate"""
        return self.e_num/self.e_den

    @property
    def d_rate(self):
        """Return decoding bits/parity rate"""
        return self.d_num/self.d_den

    def encode(self, data):
        """Encode data using specified coding"""
        data.data = self._enc.encode(data.data)
        data.PP = pproc.CODING
        data.DT = dtype.BITS
        return data

    def decode(self, data):
        """Decode data using specified coding"""
        if self.decision == Decision.SOFT:
            data.data = self._dec.decode(data.get_soft()) # TODO: implement this in the backend
        else:
            data.data = self._dec.decode(data.data)
        data.PP = pproc.CODING
        data.DT = dtype.BITS
        return data

    @staticmethod
    @abstractmethod
    def load(name: str, obj: dict):
        """Load coding from json"""

    def __rmul__(self, data):
        self._enc.log.debug(f"Coding {data}")
        data = data.as_bits()
        data = self.encode(data)
        return data

    def __rtruediv__(self, data):
        self._dec.log.debug(f"Decoding {data}")
        if isinstance(data, Mod):
            if self.decision == Decision.HARD:
                data = data.as_bits()
        data = self.decode(data) # decode may require soft/hard
        return data

class Block(Coding):
    """Parent block coding"""

    @staticmethod
    def load(name, obj):
        i_code = getattr(types, obj["type"])

        impl = type(name, (Block,), dict())
        impl.name = name
        log = Logger().Child("Coding", _config.LOG_CODING).Child(name)

        if obj["type"] == "linear":
            gen = np.array(obj["generator"], dtype=bool)
            chk = np.array(obj["check"], dtype=bool)
            i_code = i_code(log, gen, chk)


            return impl(i_code, i_code)
        if obj["type"] == "repeat":
            i_code = i_code(log, obj["count"])

            return impl(i_code, i_code)
        raise NotImplementedError(f"{name} is not implemented")

class Convolutional(Coding):
    """Parent convolutional coding"""

    @staticmethod
    def load(name, obj):
        if obj["type"] == "split":

            log = Logger().Child("Coding", _config.LOG_CODING).Child(name)

            i_e_code = getattr(types, obj["encode"]["type"])
            i_d_code = getattr(types, obj["decode"]["type"])

            i_e_num = obj["encode"]["num"]
            i_e_den = obj["encode"]["den"]
            i_e_gen = np.array(obj["encode"]["generator"], dtype=bool)
            i_e_con = obj["encode"]["constraint"]
            i_e_code = i_e_code(log, i_e_num, i_e_den, i_e_gen, i_e_con)

            i_d_num = obj["decode"]["num"]
            i_d_den = obj["decode"]["den"]
            i_d_con = obj["decode"]["constraint"]
            i_d_gen = np.array(obj["decode"]["generator"], dtype=bool)
            i_d_code = i_d_code(log, i_d_num, i_d_den, i_d_con, i_d_gen)

            impl = type(name, (Convolutional,), dict())
            impl.name = name
            return impl(i_e_code, i_d_code)
        raise NotImplementedError(f"{name} is not implemented")
