import numpy as np

from pyboiler.logger import Logger, Level

from meta import Meta
from helpers.stream import Stream
from .constellation import Constellation


class Modulation:
    __slots__ = ("log", "constellation", "map")

    def __init__(self, map: int = None):
        self.log = Logger().Child("Modulation").Child(type(self).__name__)
        self.map = map
        if self.map is not None:
            self.constellation = Constellation(type(self).points, type(self).maps[map], log=self.log)
        else:
            self.constellation = Constellation(type(self).points, log=self.log)


    def demodulate(self, fname: str, meta = Meta()):
        if self.map is None:
            raise Exception("A map must be provided to demodulate!")

        meta = Meta.from_name(fname, meta)
        symbols = np.fromfile(f"data/{fname}", getattr(np, meta.fmt))

        data = self.constellation.demodulate(symbols, meta)

        return data, meta

    def modulate(self, data: Stream, meta = Meta()):
        if self.map is None:
            raise Exception("A map must be provided to modulate!")

        symbols = self.constellation.modulate(data, meta)

        meta.mod = type(self).__name__
        return symbols, meta

class ASK(Modulation):
    pass

class FSK(Modulation):
    pass

class APSK(Modulation):
    pass

class QAM(Modulation):
    pass
