"""Coding parent classes"""
from abc import abstractmethod
from enum import Enum
import functools
import numpy as np

from ..logger import Logger
from ..errors import ConfigError

from . import base
from .. import _config
from ..base import Mod, Digital, dtype
from ..base.types import Category, Rep
from ..base.data import expect
from ..base.rpps import align, FLUSH, FRAME, STREAM

from . import types as types


class Decision(Enum):
    """Coding decision type: does the decoder get hard bits or LLRs"""
    HARD = 0
    SOFT = 1

class Coding(base.rpps.Pipe):
    """Coding parent class"""
    __slots__ = ("log", "_enc", "_dec")
    name = "Coding"
    _decision = None
    CATEGORY = Category.CODE
    # A channel code takes data bits to channel bits. A code transform is often several steps (Viterbi, deinterleave,
    # Reed-Solomon, descramble), so each step takes either side's bits; a link step then refuses channel bits
    ENCODE_TAKES, DECODE_TAKES = {Rep.DATA, Rep.CHANNEL}, {Rep.CHANNEL, Rep.DATA}
    ENCODE_GIVES, DECODE_GIVES = Rep.CHANNEL, Rep.DATA

    def takes(self, encode):
        return self.ENCODE_TAKES if encode else self.DECODE_TAKES
    def __init__(self, e_impl, d_impl):
        self.log = Logger().Child("Coding", _config.LOG_CODING).Child(type(self).__name__, _config.LOG_CODING)
        self._enc = e_impl
        self._dec = d_impl

    def __str__(self) -> str:
        if self._enc is None:
            return f"{self.name}:none"
        return f"{self.name}:{self.decision.name}:{type(self._enc).__name__}_{self.e_num}/{self.e_den}:{type(self._dec).__name__}_{self.d_num}/{self.d_den}"

    @property
    def decision(self):
        """SOFT when the decoder can use LLRs (the default for those that can), HARD otherwise. Assign to override"""
        if self._decision is not None:
            return self._decision
        return Decision.SOFT if getattr(self._dec, "soft", False) else Decision.HARD

    @decision.setter
    def decision(self, value):
        self._decision = value

    def encode(self, data):
        """Encode data using specified coding"""
        return self._run(data, True)

    def decode(self, data):
        """Decode data using specified coding

        A soft decoder takes the LLRs a demodulator produced; anything else (or data that isn't soft) gets hard bits.
        """
        return self._run(data, False)

    def _block(self, enc):
        """Items taken at a time on the way in: message bits to encode, or received bits to decode"""
        return self._enc.num if enc else self._dec.den

    # ---- header-driven control (rpps.profile.control): which definition this is, changed at a frame boundary

    CONTROLS = {"name": "live"} # Block codes and block interleavers; a code with memory restarts (Convolutional, Forney)
    folder = None # The definition kind it was loaded from ("block", "interleave"): what `control(name=...)` loads

    def control(self, name):
        """Become another definition of the same kind (`name`: "ldpc/dvbs2_64800_3_5"), or nothing at all (None: the
        bits pass as they are, an interleaver a modulation does not use). Applied between codewords, it keeps nothing
        of the old code; a profile applies it where a header says the new code starts"""
        if name is None:
            self._enc = self._dec = None
        else:
            new = _cached(self.folder or type(self).__name__.lower(), name)
            self._enc, self._dec = new._enc, new._dec
        self.name = name or "none"
        self._restart = True # A frame starts here: a codeword never spans two

    _restart = False

    def _frame_start(self, data, ctx, enc):
        """At a frame start, drop the part of a codeword the last frame left (a frame cut short), and report it"""
        if not self._restart:
            return
        from ..profile.control import of, anchor
        self._restart = False
        held = ctx.pop((self, enc), None)
        if held is not None and len(held):
            of(data.meta.stream).reports.append({"what": "cut short", "step": str(self.name), "items": int(len(held)),
                                                 "at": int(data.meta.pos)})
        anchor(data.meta, self, data.meta.pos, 0) # The frame's first item is where the output goes on from

    def pos_rate(self, decode):
        """Output bits per input bit (rpps.profile positions)"""
        if self._enc is None:
            return 1.0
        coder = self._dec if decode else self._enc
        return coder.num/coder.den if decode else coder.den/coder.num

    def _process(self, coder, x, ctx, enc):
        """Run whole blocks through the coder (ctx is this call's stream state, for coders that carry some)"""
        if not len(x):
            return np.zeros(0, dtype=bool)
        return coder.encode(x) if enc else coder.decode(x)

    def _run(self, data, enc):
        self.log.debug(f"{'Coding' if enc else 'Decoding'} {data}")
        if self._enc is None: # control(name=None): nothing to do
            return data.derive(data.data, self.CATEGORY, data.DT, type(data), dict(data.ctx), rep=data.REP)
        x = None
        if not enc and self.decision == Decision.SOFT and isinstance(data, Mod):
            x = data.get_soft()
        if x is None:
            x = data.bits()
        ctx = dict(data.ctx)
        self._frame_start(data, ctx, enc)
        x = align(ctx, (self, enc), x, self._block(enc), pad=enc, log=self.log)
        y = self._process(self._enc if enc else self._dec, x, ctx, enc)
        rep = self.ENCODE_GIVES if enc else self.DECODE_GIVES
        if np.asarray(y).dtype.kind == "f": # Soft values that went through (an interleaver): still LLRs
            return data.derive(y, self.CATEGORY, dtype.LLR, Mod, ctx, rep=rep)
        return data.derive(y, self.CATEGORY, dtype.BITS, Digital, ctx, rep=rep)

    @staticmethod
    @abstractmethod
    def load(name: str, obj: dict):
        """Load coding from json"""

    @property
    def errors(self):
        """Per codeword of the last decode: bits (Reed-Solomon: symbols) corrected, -1 where it failed. None for
        decoders without codewords (Viterbi, BCJR of a convolutional code)"""
        return getattr(self._dec, "errors", None)

    def __rmul__(self, data):
        expect(self, data, self.takes(True), True)
        return self.encode(data)

    def __rtruediv__(self, data):
        expect(self, data, self.takes(False), False)
        return self.decode(data)

    @property
    def e_num(self):
        """Return number of encoding data bits"""
        return self._enc.num
    @property
    def e_den(self):
        """Return number of total encoded bits"""
        return self._enc.den

    @property
    def d_num(self):
        """Return number of decoding data bits"""
        return self._dec.num

    @property
    def d_den(self):
        """Return number of total decoded bits"""
        return self._dec.den

    @property
    def e_rate(self):
        """Return encoding bits/parity rate"""
        return self.e_num/self.e_den

    @property
    def d_rate(self):
        """Return decoding bits/parity rate"""
        return self.d_num/self.d_den



@functools.lru_cache(maxsize=None)
def _cached(folder, name):
    """A definition loaded once per process (a DVB-S2 LDPC code takes a moment to build): what `control` swaps in"""
    from .serial import load
    return load(folder, name)

def _impl(parent, name):
    """Named subclass of `parent`, so each loaded code reports its own name"""
    impl = type(name, (parent,), dict())
    impl.name = name
    return impl

def _code_of(obj):
    try:
        return types.REGISTRY[obj["type"]]
    except KeyError:
        raise ConfigError(f"Unknown coding type {obj['type']!r}, have {sorted(types.REGISTRY)}") from None

DEFAULT_DECODER = {"linear": "trellis", "cyclic": "syndrome"} # What decodes a code type when its definition doesn't say

def _block_kernel(self, params, how):
    """The batch decoder of a code that works on whole blocks and keeps nothing between them (block codes, block
    interleavers), for the code as `params` (live parameters) make it; None where there is none (encoding, no code, a code with
    memory)"""
    if how != "/":
        return None
    if params:
        self.control(**params)
    if self._dec is None or (self.folder == "interleave" and not hasattr(self._dec, "block_bits")):
        return None
    from ..parallel.kernels import BlockDecode
    return BlockDecode(self._dec, self._block(False), self.decision == Decision.SOFT, self.CATEGORY, self.DECODE_GIVES, str(self.name))

class Block(Coding):
    SCOPE = FRAME # Whole codewords
    """Parent block coding"""
    kernel = _block_kernel

    @staticmethod
    def load(name, obj, decoder=None):
        """decoder: a registered decoder type to use instead of the code's own (it's built from the encoder),
        or "name"/"decoder" in the definition. Linear codes decode on their syndrome trellis unless that is too big"""
        log = Logger().Child("Coding", _config.LOG_CODING).Child(name)
        enc = _code_of(obj).from_config(log, obj)
        want = decoder or obj.get("decoder")
        dec = enc
        if want is None and obj["type"] in DEFAULT_DECODER:
            try:
                dec = _code_of({"type": DEFAULT_DECODER[obj["type"]]}).from_encoder(log, enc)
            except ValueError: # Too many trellis states: the encoder's own (hard) decoder it is
                pass
        elif want is not None and want != obj["type"]:
            dec = _code_of({"type": want}).from_encoder(log, enc)
        out = _impl(Block, name)(enc, dec)
        out.folder = "block"
        return out

class Convolutional(Coding):
    """Parent convolutional coding: separate encoder ("encode") and decoder ("decode") implementations"""
    CONTROLS = {"name": "restart"} # The path metrics belong to the old trellis
    folder = "conv"

    @staticmethod
    def load(name, obj, decoder=None):
        """decoder: use this decoder type (viterbi, bcjr) with the definition's decode parameters"""
        if obj["type"] != "split":
            raise ConfigError(f"{name}: a convolutional definition is type 'split' (encode + decode), not {obj['type']!r}")
        log = Logger().Child("Coding", _config.LOG_CODING).Child(name)
        e, d = dict(obj["encode"]), dict(obj["decode"])
        if decoder is not None:
            d["type"] = decoder
        for side in (e, d): # puncture / termination / constraint are shared between both sides
            side.setdefault("puncture", obj["encode"].get("puncture"))
            for k in ("termination", "terminated", "constraint"):
                if k in obj["encode"]:
                    side.setdefault(k, obj["encode"][k])
        return _impl(Convolutional, name)(_code_of(e).from_config(log, e), _code_of(d).from_config(log, d))

    def _block(self, enc):
        # A punctured code works on whole puncture periods
        coder = self._enc if enc else self._dec
        p = getattr(coder, "puncture", None)
        if p is None:
            return coder.num if enc else coder.den
        return coder.num*p.shape[1] if enc else int(p.sum())

    def _process(self, coder, x, ctx, enc):
        """The register / path metrics go on from the previous call; a flush ends the frame (terminated code: its tail)"""
        key = (self, enc, "state")
        final = bool(ctx.get(FLUSH))
        if getattr(coder, "stream", True):
            st = dict(ctx.get(key, {}))
            y = coder.encode(x, st, final) if enc else coder.decode(x, state=st, final=final)
            if st and not final:
                ctx[key] = st
            else:
                ctx.pop(key, None)
            return y
        # A decoder of whole frames (bcjr) collects the stream and decodes it at the end
        prev = ctx.pop(key, None)
        if prev is not None:
            x = np.concatenate((prev, x))
        if not final:
            ctx[key] = x
            return np.zeros(0, dtype=bool)
        if not len(x):
            return np.zeros(0, dtype=bool)
        return coder.encode(x) if enc else coder.decode(x)

class Interleaver(Coding):
    SCOPE = FRAME # A block interleaver; a Forney one carries delay lines (STREAM, set when loaded)
    kernel = _block_kernel
    """Interleaver: Forney (convolutional: I branches, M symbols of delay increment, state carried along the stream)
    or block (a permutation of each block)"""

    @staticmethod
    def load(name, obj, decoder=None):
        log = Logger().Child("Coding", _config.LOG_CODING).Child(name)
        code = _code_of(obj).from_config(log, obj)
        out = _impl(Interleaver, name)(code, code)
        out.folder = "interleave"
        if not hasattr(code, "block_bits"):
            out.CONTROLS = {"name": "restart"} # Its delay lines hold data interleaved the old way
            out.SCOPE = STREAM
        return out

    def pos_rate(self, decode):
        return 1.0

    def _block(self, enc):
        if hasattr(self._enc, "block_bits"):
            return self._enc.block_bits
        return self._enc.I*self._enc.symbol # Whole rows of I symbols: the first one is on branch 0

    def _process(self, coder, x, ctx, enc):
        if not len(x):
            return np.zeros(0, dtype=bool)
        if hasattr(coder, "block_bits"): # A block interleaver keeps nothing between blocks
            return coder.encode(x) if enc else coder.decode(x)
        key = (self, enc, "delay")
        ctx[key] = state = dict(ctx.get(key, {})) # The delay lines go on from the previous call
        return coder.encode(x, state=state) if enc else coder.decode(x, state=state)
