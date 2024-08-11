from pyboiler.logger import Logger, Level

from .helpers import Stream
from .helpers import bitarray
from .meta import Meta

from .mod import Modulation
from .coding import Coding
import numpy as np

'''
makes an rrc filter at 4 sps
'''
def rrc_filter(n=1000000,beta=0.2):
    sps = 4.0
    tmat = np.arange(n) / sps
    t = tmat - 16 - 1e-9
    sin_arg = np.sin(np.pi * t * (1.0 - beta))
    cos_arg = 4.0 * beta * t * np.cos(np.pi * t * (1 + beta))
    denom = np.pi * t * (1.0 - 16.0 * (beta * t) * (beta * t))
    filt = (sin_arg + cos_arg) / denom
    filt[::2] *= -1
    return np.fft.fft(filt)


def upsample(sig, over):
    assert len(sig) <= 250000
    # first go to four sps
    signal = np.zeros(1000000, dtype=np.complex128)
    signal[:len(sig)*4:4] = sig

    signal[::2] *= -1
    den = 1000000
    num = int((over*den)) + 3
    # make it divesable by four
    num -= num % 4
    f = np.fft.fft(signal, n=den) * rrc_filter(n=den)

    ret = np.zeros(num, dtype=np.complex64)
    ret[:den] = f
    ret = np.roll(ret, -den//2)
    ret = np.fft.ifft(ret)[::4]
    retlen = int((len(sig) + 1) * over)

    return ret[:retlen]

class Pipeline:
    def __init__(self, mod, ecc=None, sps=1.0):
        self.log = Logger().Child("Pipeline", Level.WARN)
        self.meta = Meta()
        self.mod: Modulation = mod
        self.coding: Coding = ecc  # type: ignore
        self.sps = sps # Should maybe be a meta??

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
        if self.sps > 1.0:
            # upsampleing
            syms = upsample(syms, self.sps)



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
