"""Modulation parent classes"""
from abc import abstractmethod
import numpy as np

from pyboiler.logger import Logger, Level

from . import base
from . import _config
from ..base import Analog, Mod, Digital, pproc, dtype
from . import lfsr


class Scram(base.rpps.Pipe):
    """Scram Pipe"""

    def __init__(self):
        self.log = Logger().Child("Scram", _config.LOG_SCRAM).Child(type(self).__name__, _config.LOG_SCRAM)

    def __str__(self):
        return f"{type(self).__name__}"

    @abstractmethod
    def scram(self, data):
        """Encode data using specified scram"""

    @abstractmethod
    def descram(self, data):
        """Decode data using specified scram"""

    @staticmethod
    @abstractmethod
    def load(name: str, obj: dict):
        """Load modulation from json"""

    def __rmul__(self, data):
        self.log.debug(f"Scrambling {data}")
        data = data.as_bits()
        data.data = self.scram(data.data)
        data.PP = pproc.SCRAM
        data.DT = dtype.BITS
        return data

    def __rtruediv__(self, data):
        self.log.debug(f"Descrambling {data}")
        data = data.as_bits()
        data.data = self.descram(data.data)
        data.PP = pproc.SCRAM
        data.DT = dtype.BITS
        return data


class Feedthrough(Scram):
    """Feedthrough scrambler"""
    def __init__(self, scram_lfsr: lfsr.LFSR, descram_lfsr: lfsr.LFSR):
        super().__init__()
        self.s_lfsr = scram_lfsr
        self.d_lfsr = descram_lfsr
    def __str__(self):
        return f"{type(self).__name__}:{self.s_lfsr}"

    def reset(self):
        """Reset LFSR"""
        self.s_lfsr.reset()
        self.d_lfsr.reset()

    def scram(self, data):
        scr_data = np.empty_like(data, dtype=bool)

        for i, bit in enumerate(data):
            scr_data[i] = self.s_lfsr.get_bit() ^ bit

        return scr_data

    def descram(self, data):
        scr_data = np.empty_like(data, dtype=bool)

        for i, bit in enumerate(data):
            scr_data[i] = self.d_lfsr.get_bit() ^ bit

        return scr_data

    @staticmethod
    def load(name, obj):
        i_lfsr = getattr(lfsr, obj["type"])
        i_seed = np.array(obj["seed"], dtype=bool)
        i_poly = np.array(obj["poly"], dtype=int)
        i_s_lfsr = i_lfsr(np.copy(i_seed), np.copy(i_poly))
        i_d_lfsr = i_lfsr(np.copy(i_seed), np.copy(i_poly))
        impl = type(name, (Feedthrough,), dict())(i_s_lfsr, i_d_lfsr)
        return impl


class Additive(Scram):

    @staticmethod
    def load(name, obj):
        impl = type(name, (Additive,), dict(name=name, poly=obj["poly"]))()
        return impl
