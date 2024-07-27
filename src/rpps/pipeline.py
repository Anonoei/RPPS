from pyboiler.logger import Logger, Level

from .helpers import Stream
from .helpers import bitarray
from .meta import Meta

from .mod import Modulation
from .coding import Coding

class Pipeline:
    def __init__(self, mod, ecc=None):
        self.log = Logger().Child("Pipeline", Level.TRACE)
        self.meta = Meta()
        self.mod: Modulation = mod
        self.coding: Coding = ecc  # type: ignore

        if self.mod is not None:
            self.mod.init_meta(self.meta)
        if self.coding is not None:
            self.coding.init_meta(self.meta)

    def __str__(self):
        return self.meta.short()

    def __repr__(self):
        return f"<Pipeline {str(self.meta)}>"

    def enc(self, data: Stream):
        self.log.trace(f"Encoding {repr(self)} {data}")
        if isinstance(data, bytes):
            data = Stream.from_bytes(data)

        if issubclass(type(self.coding), Coding):
            encoded = self.coding.encode(data)
            data = Stream.from_bytes(encoded.to_bytes())

        syms, meta = self.mod.modulate(data, self.meta)
        self.meta = meta
        return syms

    def dec(self, symbols, meta):

        data, meta = self.mod.demodulate(symbols, meta)
        self.meta = meta

        if issubclass(type(self.coding), Coding):
            decoded = self.coding.decode(data)
            data = Stream.from_bytes(decoded.to_bytes())
        self.log.trace(f"Decoded {data.bytes}")
        return data

    def from_file(self, path):
        meta, symbols = Meta.from_file(path)
        return self.dec(symbols, meta)
