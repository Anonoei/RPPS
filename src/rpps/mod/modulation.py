import numpy as np

from pyboiler.logger import Logger, Level

from . import Meta
from . import Stream

from .constellation import Maps, Points
from .constellation import Constellation


class Modulation:
    __slots__ = ("log", "constellation", "map")
    name = "Modulation"
    points = Points([])
    maps = Maps([])

    def __init__(self, map_idx: int = None):  # type: ignore
        self.log = Logger().Child("Modulation").Child(type(self).name)
        self.map = map_idx
        if self.map is not None:
            self.constellation = Constellation(
                type(self).points,
                type(self).maps[self.map],
                log=self.log)
        else:
            self.constellation = Constellation(
                type(self).points,
                log=self.log)

    def demodulate(self, symbols: np.ndarray, meta=Meta()):
        if self.map is None:
            raise Exception("A map must be provided to demodulate!")

        data = self.constellation.demodulate(symbols, meta)

        return data, meta

    def modulate(self, data: Stream, meta=Meta()):
        if self.map is None:
            raise Exception("A map must be provided to modulate!")

        symbols = self.constellation.modulate(data, meta)

        meta.mod = type(self).name
        return symbols, meta


class ASK(Modulation):
    pass


class FSK(Modulation):
    pass


class APSK(Modulation):
    pass


class QAM(Modulation):
    pass
