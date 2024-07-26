from pyboiler.logger import Logger, Level

from . import Stream
from . import bitarray
from ..meta import Meta

from ..mod import Modulation
from ..coding import Coding

class Pipeline:
    def __init__(self, mod, ecc=None):
        self.log = Logger().Child("Pipeline", Level.TRACE)
        self.mod: Modulation = mod
        self.ecc: Coding = ecc  # type: ignore

    def __repr__(self):
        return f"<Pipeline {type(self.mod).__name__}-{type(self.ecc).__name__}>"

    def enc(self, data: Stream):
        self.log.trace(f"Encoding {repr(self)} {data}")
        if isinstance(data, bytes):
            data = Stream.from_bytes(data)

        if issubclass(type(self.ecc), Coding):
            encoded = self.ecc.encode(data)
            data = Stream.from_bytes(encoded.to_bytes())

        syms, meta = self.mod.modulate(data)
        return syms, meta

    def dec(self, symbols, meta):

        data, meta = self.mod.demodulate(symbols, meta)

        if issubclass(type(self.ecc), Coding):
            decoded = self.ecc.decode(data)
            data = Stream.from_bytes(decoded.to_bytes())
        self.log.trace(f"Decoded {data.bytes}")
        return data, meta

    def from_file(self, path):
        meta, symbols = Meta.from_file(path)
        return self.dec(symbols, meta)
