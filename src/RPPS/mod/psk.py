from .mod import Modulation, np

from meta import Meta
from .constellation import Constellation

from pyboiler.logger import Logger, Level

Logger().Modulation.Child("PSK")

class PSK(Modulation):
    def __init__(self):
        self.log = Logger().Child("Modulation").Child(type(self).__name__)

    def process(self, fname: str, meta = Meta()):
        meta = Meta.from_name(fname, meta)
        symbols = np.fromfile(f"data/{fname}", getattr(np, meta.fmt))

        data = self.constellation.decode(symbols, meta)

        return data, meta

    def generate(self, data: bytes, meta = Meta()):
        symbols = self.constellation.encode(data, meta)

        meta.mod = type(self).__name__
        return symbols, meta
