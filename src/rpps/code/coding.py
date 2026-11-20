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
from ..base.rpps import align, FLUSH, FRAME, STREAM, failed_of
from ..base.control import of, anchor

from . import types as types


class Decision(Enum):
    """Coding decision type: does the decoder get hard bits or LLRs"""
    HARD = 0
    SOFT = 1

class Coding(base.rpps.Pipe):
    """Coding parent class"""
    __slots__ = ("log", "encoder", "decoder")
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
        self.encoder = e_impl
        self.decoder = d_impl

    def __str__(self) -> str:
        if self.encoder is None:
            return f"{self.name}:none"
        return f"{self.name}:{self.decision.name}:{type(self.encoder).__name__}_{self.e_num}/{self.e_den}:{type(self.decoder).__name__}_{self.d_num}/{self.d_den}"

    @property
    def decision(self):
        """SOFT when the decoder can use LLRs (the default for those that can), HARD otherwise. Assign to override"""
        if self._decision is not None:
            return self._decision
        return Decision.SOFT if getattr(self.decoder, "soft", False) else Decision.HARD

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
        return self.encoder.num if enc else self.decoder.den

    # ---- header-driven control (rpps.profile.control): which definition this is, changed at a frame boundary

    CONTROLS = {"name": "live"} # Block codes and block interleavers; a code with memory restarts (Convolutional, Forney)
    folder = None # The definition kind it was loaded from ("block", "interleave"): what `control(name=...)` loads

    def control(self, name):
        """Become another definition of the same kind (`name`: "ldpc/dvbs2_64800_3_5"), or nothing at all (None: the
        bits pass as they are, an interleaver a modulation does not use). Applied between codewords, it keeps nothing
        of the old code; a profile applies it where a header says the new code starts"""
        if name is None:
            self.encoder = self.decoder = None
        else:
            new = cached(f"{self.folder or type(self).__name__.lower()}/{name}")
            self.encoder, self.decoder = new.encoder, new.decoder
        self.name = name or "none"
        self._restart = True # A frame starts here: a codeword never spans two

    _restart = False

    def _frame_start(self, data, ctx, enc):
        """At a frame start, drop the part of a codeword the last frame left (a frame cut short), and report it"""
        if not self._restart:
            return
        self._restart = False
        held = ctx.pop((self, enc), None)
        if held is not None and len(held):
            of(data.meta.stream).reports.append({"what": "cut short", "step": str(self.name), "items": int(len(held)),
                                                 "at": int(data.meta.pos)})
        anchor(data.meta, self, data.meta.pos, 0) # The frame's first item is where the output goes on from

    def pos_rate(self, decode):
        """Output bits per input bit (rpps.profile positions)"""
        if self.encoder is None:
            return 1.0
        coder = self.decoder if decode else self.encoder
        return coder.num/coder.den if decode else coder.den/coder.num

    def _process(self, coder, x, ctx, enc):
        """Run whole blocks through the coder (ctx is this call's stream state, for coders that carry some)"""
        if not len(x):
            return np.zeros(0, dtype=bool)
        return coder.encode(x) if enc else coder.decode(x)

    def _run(self, data, enc):
        self.log.debug(f"{'Coding' if enc else 'Decoding'} {data}")
        if self.encoder is None: # control(name=None): nothing to do
            return data.derive(data.data, self.CATEGORY, data.DT, type(data), dict(data.ctx), rep=data.REP)
        x = None
        if not enc and self.decision == Decision.SOFT and isinstance(data, Mod):
            x = data.get_soft()
        if x is None:
            x = data.bits()
        ctx = dict(data.ctx)
        self._frame_start(data, ctx, enc)
        x = align(ctx, (self, enc), x, self._block(enc), pad=enc, log=self.log)
        y = self._process(self.encoder if enc else self.decoder, x, ctx, enc)
        rep = self.ENCODE_GIVES if enc else self.DECODE_GIVES
        if np.asarray(y).dtype.kind == "f": # Soft values that went through (an interleaver): still LLRs
            out = data.derive(y, self.CATEGORY, dtype.LLR, Mod, ctx, rep=rep)
        else:
            out = data.derive(y, self.CATEGORY, dtype.BITS, Digital, ctx, rep=rep)
        out.failed = None if enc else self._failed(failed_of(data), len(data), len(x), len(y))
        return out

    def _failed(self, prev, n_data, n_in, n):
        """The `failed` mask of a decode's output (None: all decoded): its own verdict per codeword when it has one (a later
        code's success clears an earlier failure: BCH after LDPC), else what came in failed, carried block by block through
        a permutation (an interleaver); nothing known otherwise. prev: the input's mask over n_data items, of which the
        decoder took n_in (whole blocks, the same items when nothing was held back)"""
        if not n:
            return None
        errs = getattr(self.decoder, "errors", None)
        mask = None
        if errs is not None and len(errs) and n % len(errs) == 0:
            mask = np.repeat(np.asarray(errs) < 0, n//len(errs))
        elif prev is not None and n_data == n_in == n:
            blk = self._block(False)
            if blk and n % blk == 0:
                mask = np.repeat(np.asarray(prev).reshape(-1, blk).any(axis=1), blk)
        return mask if mask is not None and mask.any() else None

    @staticmethod
    @abstractmethod
    def load(name: str, obj: dict):
        """Load coding from json"""

    @property
    def errors(self):
        """Per codeword of the last decode: bits (Reed-Solomon: symbols) corrected, -1 where it failed. None for
        decoders without codewords (Viterbi, BCJR of a convolutional code)"""
        return getattr(self.decoder, "errors", None)


    @property
    def e_num(self):
        """Return number of encoding data bits"""
        return self.encoder.num
    @property
    def e_den(self):
        """Return number of total encoded bits"""
        return self.encoder.den

    @property
    def d_num(self):
        """Return number of decoding data bits"""
        return self.decoder.num

    @property
    def d_den(self):
        """Return number of total decoded bits"""
        return self.decoder.den

    @property
    def e_rate(self):
        """Return encoding bits/parity rate"""
        return self.e_num/self.e_den

    @property
    def d_rate(self):
        """Return decoding bits/parity rate"""
        return self.d_num/self.d_den



def puncture_period(code):
    """Channel bits in one period of a convolutional code's puncture pattern (n, the outputs per step, when unpunctured):
    the alignments a decoder can start at. `code`: anything with an `encoder` (a loaded Coding)"""
    p = getattr(code.encoder, "puncture", None)
    return int(p.sum()) if p is not None else int(code.encoder.den)

def mismatch(code, hard):
    """Decode hard channel bits, encode the decision again and compare: (bits that differ, bits compared). At the right
    code and alignment only channel errors differ; a wrong one differs about as often as random bits"""
    hard = np.asarray(hard)
    e = code.encoder.encode(code.decoder.decode(hard.astype(np.uint8)).astype(bool))
    n = min(len(e), len(hard))
    return int(np.count_nonzero(e[:n] != hard[:n].astype(bool))), n

def mismatch_batch(code, hard):
    """mismatch() of each row of hard (W, L), decoded together (decode_batch where the decoder has it): (differing
    bits (W,), bits compared)"""
    hard = np.asarray(hard).astype(np.uint8)
    dec = code.decoder
    u = dec.decode_batch(hard) if hasattr(dec, "decode_batch") else np.array([dec.decode(h) for h in hard])
    enc = code.encoder
    e = enc.encode_batch(u) if hasattr(enc, "encode_batch") else np.array([enc.encode(np.asarray(x).astype(bool)) for x in u])
    n = min(e.shape[1], hard.shape[1])
    return np.count_nonzero(e[:, :n] != hard[:, :n].astype(bool), axis=1).astype(np.int64), n

@functools.lru_cache(maxsize=None)
def cached(ref):
    """A definition loaded once per process and shared (a DVB-S2 LDPC code takes a moment to build): what `control`
    swaps in, and what the blind tools try: `rp.code.cached("conv/k7_171_133/3_4")`"""
    from .serial import load
    return load(ref)

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
    if self.decoder is None or (self.folder == "interleave" and not hasattr(self.decoder, "block_bits")):
        return None
    from .kernels import BlockDecode
    return BlockDecode(self.decoder, self._block(False), self.decision == Decision.SOFT, self.CATEGORY, self.DECODE_GIVES, str(self.name))

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
        """{"base": "conv", "constraint": K, "generator": [...], "puncture": [...], "decoder": "viterbi"}: one set of keys
        builds the encoder and its decoder. decoder: use this decoder type (viterbi, bcjr) instead of the definition's"""
        log = Logger().Child("Coding", _config.LOG_CODING).Child(name)
        if "encode" in obj or "decode" in obj:
            raise ConfigError(f"{name}: a convolutional code is one set of keys (constraint, generator, puncture, decoder), "
                              "not separate encode / decode blocks")
        if "generator" not in obj:
            raise ConfigError(f"{name}: a convolutional code needs 'generator' (and 'constraint')")
        keys = {k: v for k, v in obj.items() if k not in ("base", "type", "comment", "ref", "tags", "params", "decoder")}
        keys.setdefault("num", 1)
        keys.setdefault("den", len(obj["generator"]))
        e = {**keys, "type": "conv"}
        d = {**keys, "type": decoder or obj.get("decoder", "viterbi")}
        return _impl(Convolutional, name)(_code_of(e).from_config(log, e), _code_of(d).from_config(log, d))

    def _block(self, enc):
        # A punctured code works on whole puncture periods
        coder = self.encoder if enc else self.decoder
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
        if enc:
            return coder.encode(x)
        # A soft-output decoder (BCJR) gives its a-posteriori LLRs on: the next code step (an outer Reed-Solomon code's
        # erasures, an iterative receiver) still has the soft values
        return coder.app(x) if hasattr(coder, "app") else coder.decode(x)

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
        if hasattr(self.encoder, "block_bits"):
            return self.encoder.block_bits
        return self.encoder.I*self.encoder.symbol # Whole rows of I symbols: the first one is on branch 0

    def _process(self, coder, x, ctx, enc):
        if not len(x):
            return np.zeros(0, dtype=bool)
        if hasattr(coder, "block_bits"): # A block interleaver keeps nothing between blocks
            return coder.encode(x) if enc else coder.decode(x)
        key = (self, enc, "delay")
        ctx[key] = state = dict(ctx.get(key, {})) # The delay lines go on from the previous call
        return coder.encode(x, state=state) if enc else coder.decode(x, state=state)
