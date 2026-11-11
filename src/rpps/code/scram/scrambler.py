"""Scramblers: how a sequence is applied to bits (the `scramble` resource; the sequences are rpps.sequence)"""
from abc import abstractmethod
import numpy as np

from ...logger import Logger

from . import base
from . import _config
from ...base import Analog, Mod, Digital, dtype
from ...base.types import Category, Rep
from ...base.data import expect
from ...errors import ConfigError
from ...sequence import lfsr


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
        return data.derive(y, Category.CODE, dtype.BITS, Digital, ctx, rep=Rep.CHANNEL if enc else Rep.DATA)

    def _stream(self, ctx, key, bits, enc):
        """Scramble/descramble bits continuing from ctx[key] (the previous call), and put the new state back"""
        raise NotImplementedError()

    CATEGORY = Category.CODE

    def takes(self, encode):
        return {Rep.DATA, Rep.CHANNEL} # One step of the code transform, like Coding

    def pos_rate(self, decode):
        return 1.0

    # ---- header-driven control (rpps.profile.control)

    CONTROLS = {"length": "live"}
    length = None # Bits per frame, when a profile controls it: the sequence restarts at every frame start
    _restart = False

    def control(self, length=None):
        """A frame of `length` bits starts here (DVB-S2: a BBFRAME of Kbch bits): an additive sequence restarts from its
        seed, here and every `length` bits after. A self-synchronizing scrambler has nothing to restart"""
        self.length = None if length is None else int(length)
        self._restart = True

    def _frame_start(self, ctx, key):
        """Restart the sequence where a frame starts (control), and every `length` bits after it"""
        if self._restart:
            self._restart = False
            ctx[key] = 0

    def __rmul__(self, data):
        expect(self, data, self.takes(True), True)
        return self._run(data, True)

    def __rtruediv__(self, data):
        expect(self, data, self.takes(False), False)
        return self._run(data, False)

    @staticmethod
    @abstractmethod
    def load(name: str, obj: dict):
        """Load modulation from json"""



class Feedthrough(Scram):
    """Feedthrough (self-synchronizing) scrambler: the scrambled bits go through the register, so the keystream depends on the
    data. y[n] = d[n] ^ y[n-t] for each tap t; descrambling is d[n] = y[n] ^ y[n-t], which needs no seed once a register
    length of bits has passed (a bit slip or a mid-stream start is lost after that), and turns one bit error into one per tap"""
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
    def _run_bits(data, reg, descramble):
        out = np.empty_like(data, dtype=bool)
        for i, bit in enumerate(data):
            out[i] = reg.advance(bit, descramble)
        return out

    def encode(self, data):
        """Scramble, continuing from where the last call left this scrambler"""
        return self._run_bits(data, self.s_lfsr, False)

    def decode(self, data):
        return self._run_bits(data, self.d_lfsr, True)

    def _stream(self, ctx, key, bits, enc):
        reg = ctx[key].clone() if key in ctx else (self.s_lfsr if enc else self.d_lfsr).fresh()
        out = self._run_bits(bits, reg, not enc)
        ctx[key] = reg
        return out

    @staticmethod
    def load(name, obj):
        i_lfsr = getattr(lfsr, obj["register"])
        i_seed = np.array(obj["seed"], dtype=bool)
        i_poly = np.array(obj["poly"], dtype=int)
        i_s_lfsr = i_lfsr(np.copy(i_seed), np.copy(i_poly))
        i_d_lfsr = i_lfsr(np.copy(i_seed), np.copy(i_poly))
        impl = type(name, (Feedthrough,), dict())(i_s_lfsr, i_d_lfsr)
        return impl


class Additive(Scram):
    """Additive (synchronous) scrambler: data XOR a sequence that does not depend on the data (rpps.sequence)

    encode(bits, start) and decode (the same) XOR the sequence from position `start`; soft(llr, start) flips LLRs instead.
    As an operator (`bits * scr`, `bits / scr`) its position carries along the stream, and restarts:
        period: every `period` bits (an LTE PBCH over four radio frames, a P25 phase 2 superframe)
        frame: packet structure for byte streams, as in DVB energy dispersal:
            bytes: restart period; sync_period/sync_bytes: leading bytes of each packet the sequence skips over (it
            still advances, those bytes just aren't touched); skip_first: bytes before it starts (default sync_bytes);
            marker: the first byte of each frame as sent (DVB: the inverted sync byte, 0xB8), written on encode, and
            `sync` (default marker ^ 0xFF) written on decode. The stream must start on a frame boundary.
        length (a profile's control): every `length` bits from where a frame starts (a DVB-S2 BBFRAME)
    """
    def __init__(self, sequence, frame=None, period=None):
        super().__init__()
        self.seq = sequence
        self.frame = frame
        self.period = None if period is None else int(period)
        self.marker = None if frame is None else frame.get("marker")
        self._mask = None

    def __str__(self):
        return f"{type(self).__name__}:{self.seq.name}"

    def mask(self):
        """Bits XOR'd onto one frame (frame bytes * 8), or onto one period"""
        if self._mask is None:
            self._mask = self._make_mask() if self.frame is not None else self.seq.bits(self.period)
        return self._mask

    def _make_mask(self):
        n = self.frame["bytes"]
        sp, sb = self.frame.get("sync_period", 0), self.frame.get("sync_bytes", 0)
        first = self.frame.get("skip_first", sb) # The sequence starts after the first sync byte
        seq = self.seq.bits((n - first)*8).reshape(-1, 8)
        mask = np.zeros((n, 8), dtype=np.uint8)
        mask[first:] = seq
        if sp:
            idx = np.arange(n)
            mask[(idx % sp) < sb] = 0
        return mask.reshape(-1)

    def _bits(self, n, start):
        if self.frame is None and self.period is None:
            return self.seq.bits(n, start)
        m = self.mask()
        return m[(start + np.arange(n)) % len(m)]

    def encode(self, data, start=0):
        """data XOR the sequence from position `start` (0: a frame or period boundary)"""
        bits = np.asarray(data).astype(np.uint8)
        return (bits ^ self._bits(len(bits), start)).astype(bool)

    decode = encode

    def soft(self, llr, start=0):
        """Descramble LLRs (positive = 0): the sign flips where the sequence is 1"""
        llr = np.asarray(llr, dtype=np.float64)
        return llr*(1.0 - 2.0*self._bits(len(llr), start))

    def _stream(self, ctx, key, bits, enc):
        self._frame_start(ctx, key)
        if self.length:
            return _framed(self, ctx, key, bits)
        start = ctx.get(key, 0)
        out = self.encode(bits, start)
        if self.marker is not None and len(bits): # The first byte of each frame: the marker as sent, the sync byte as read
            pos = (start + np.arange(len(bits))) % len(self.mask())
            first = pos < 8
            if first.any():
                byte = self.marker if enc else self.frame.get("sync", self.marker ^ 0xFF)
                out[first] = ((byte >> (7 - pos[first])) & 1).astype(bool)
        end = start + len(bits)
        ctx[key] = end % len(self.mask()) if (self.frame is not None or self.period) else end
        return out

    @staticmethod
    def load(name, obj):
        from ... import sequence as _sequence
        ref = obj.get("sequence")
        if ref is None:
            raise ConfigError(f"scramble {name}: an additive scrambler names its 'sequence'")
        seq = _sequence.load(ref) if isinstance(ref, str) else _sequence.load(ref["name"], **ref.get("args", {}))
        return type(name, (Additive,), dict(name=name))(seq, obj.get("frame"), obj.get("period"))


def _framed(scr, ctx, key, bits):
    """An additive scrambler restarted every `scr.length` bits: ctx[key] is the position in the current frame"""
    bits = np.asarray(bits)
    pos = ctx.get(key, 0)
    out, i = [], 0
    while i < len(bits):
        n = min(len(bits) - i, scr.length - pos)
        out.append(scr.encode(bits[i:i + n], pos))
        i, pos = i + n, pos + n
        if pos >= scr.length:
            pos = 0
    ctx[key] = pos
    return np.concatenate(out) if out else np.zeros(0, dtype=bool)
