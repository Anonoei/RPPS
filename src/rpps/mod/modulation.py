from pyboiler.logger import Logger

from . import base
from . import _config

from ..base import Analog, Mod, Digital

class Modulation(base.rpps.Pipe):
    """Modulation parent class"""
    __slots__ = ("log",)
    name = "Modulation"

    def __init__(self):
        self.log = Logger().Child("Modulation", _config.LOG_MOD).Child(type(self).name)
    def __str__(self):
        return f"{type(self).__name__}"

    def encode(self, data: Digital) -> Mod:
        """Convert bits to IQ samples"""
        raise NotImplementedError()
    def decode(self, data: Analog) -> Mod:
        """Convert IQ samples to bits"""
        raise NotImplementedError()

    def __rmul__(self, data):
        self.log.debug(f"Modulating {data}")
        return self.encode(data)

    def __rtruediv__(self, data):
        self.log.debug(f"Demodulating {data}")
        return self.decode(data)

    def draw_refs(self, points: bool = True, ref: bool = True, ax=None):
        """Draw constellation points on viz"""
        return NotImplemented

    @staticmethod
    def load(name: str, obj: dict):
        """Load modulation from json"""
        raise NotImplementedError()
