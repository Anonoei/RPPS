"""Modulation parent classes"""
from abc import abstractmethod
import numpy as np

from ..logger import Logger

from . import base
from . import _config
from ..base import Analog, Mod, Digital, pproc, dtype
from . import lfsr


class Scram(base.rpps.Pipe):
    """Scrambler parent class"""

    def __init__(self):
        self.log = Logger().Child("Scram", _config.LOG_SCRAM).Child(type(self).__name__, _config.LOG_SCRAM)

    def __str__(self):
        return f"{type(self).__name__}"

    def _run(self, data, enc):
        """Scramble or descramble. The generator state lives in the stream (data.ctx): new data starts at the seed"""
        self.log.debug(f"{'Scrambling' if enc else 'Descrambling'} {data}")
        ctx = dict(data.ctx)
        y = self._stream(ctx, (self, enc), data.bits(), enc)
        return data.derive(y, pproc.SCRAM, dtype.BITS, Digital, ctx)

    def _stream(self, ctx, key, bits, enc):
        """Scramble/descramble bits continuing from ctx[key] (the previous call), and put the new state back"""
        raise NotImplementedError()

    def __rmul__(self, data):
        return self._run(data, True)

    def __rtruediv__(self, data):
        return self._run(data, False)

    @staticmethod
    @abstractmethod
    def load(name: str, obj: dict):
        """Load modulation from json"""



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

    @staticmethod
    def _xor(data, reg):
        scr_data = np.empty_like(data, dtype=bool)
        for i, bit in enumerate(data):
            scr_data[i] = reg.get_bit() ^ bit
        return scr_data

    def encode(self, data):
        """Scramble, continuing from where the last call left this scrambler"""
        return self._xor(data, self.s_lfsr)

    def decode(self, data):
        return self._xor(data, self.d_lfsr)

    def _stream(self, ctx, key, bits, enc):
        reg = ctx[key].clone() if key in ctx else (self.s_lfsr if enc else self.d_lfsr).fresh()
        out = self._xor(bits, reg)
        ctx[key] = reg
        return out

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
    """Additive (synchronous) scrambler: data xor a PRBS that does not depend on the data

    poly: feedback taps (register positions, 1 = first stage), seed: register bits, first stage first
    frame: optional packet structure for byte streams, as in DVB energy dispersal:
        bytes: PRBS restart period; sync_period/sync_bytes: leading bytes of each packet that the PRBS skips over
        (it still advances, the data bytes just aren't touched)
    encode/decode take bits that start on a frame boundary; both are the same operation.
    """
    def __init__(self, poly, seed, frame=None):
        super().__init__()
        self.poly = np.asarray(poly, dtype=int)
        self.seed = np.asarray(seed, dtype=np.uint8)
        self.frame = frame
        self.marker = None if frame is None else frame.get("marker")
        self._mask = None

    def sequence(self, nbits):
        """PRBS bits from the seed (Fibonacci register)"""
        r = self.seed.copy()
        out = np.zeros(nbits, dtype=np.uint8)
        taps = self.poly - 1
        for i in range(nbits):
            b = np.bitwise_xor.reduce(r[taps]) & 1
            out[i] = b
            r[1:] = r[:-1]
            r[0] = b
        return out

    def mask(self):
        """Bits to xor onto one frame (length frame bytes * 8), or one PRBS period without a frame"""
        if self._mask is None:
            self._mask = self._make_mask()
        return self._mask

    def _make_mask(self):
        if self.frame is None:
            return self.sequence(int(2**len(self.seed) - 1))
        n = self.frame["bytes"]
        sp, sb = self.frame.get("sync_period", 0), self.frame.get("sync_bytes", 0)
        first = self.frame.get("skip_first", sb) # The PRBS starts after the first sync byte
        seq = self.sequence((n - first)*8).reshape(-1, 8)
        mask = np.zeros((n, 8), dtype=np.uint8)
        mask[first:] = seq
        if sp:
            idx = np.arange(n)
            mask[(idx % sp) < sb] = 0
        return mask.reshape(-1)

    def encode(self, data, start=0):
        """start: position in the mask the first bit falls on (0: a frame boundary)"""
        bits = np.asarray(data).astype(np.uint8)
        m = np.roll(self.mask(), -start)
        reps = -(-len(bits)//len(m))
        return (bits ^ np.tile(m, reps)[:len(bits)]).astype(bool)

    decode = encode

    def _stream(self, ctx, key, bits, enc):
        start = ctx.get(key, 0)
        out = self.encode(bits, start)
        ctx[key] = (start + len(bits)) % len(self.mask())
        return out

    @staticmethod
    def load(name, obj):
        seed = obj["seed"]
        if isinstance(seed, str):
            seed = [int(c) for c in seed]
        return type(name, (Additive,), dict(name=name))(obj["poly"], seed, obj.get("frame"))
