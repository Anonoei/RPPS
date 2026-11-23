"""A sync marker among channel bits: a known word before each block of `length` bits (CCSDS ASM before each frame)

    coded = frames * rp.code.Marker(0x1ACFFC1D, 32, length=8160)     # transmit: the word, then 8160 bits, again...
    frames = aligned / rp.code.Marker(0x1ACFFC1D, 32, length=8160)   # receive: the word stripped, mismatches counted

Attaching the marker is a code transform (known structure among channel bits); finding it in an unaligned stream is sync:
`rp.sync.Delimiter(word, bits, length=..., keep=True)` gives each frame from its word on, aligned, which this then strips.
In a profile: {"sync": "delimit", "word": "0x1ACFFC1D", "bits": 32, "length": 8160, "keep": true} then
{"code": "marker", "word": "0x1ACFFC1D", "bits": 32, "length": 8160}; or both name a `sequence` of type "word".
Without `length` the frames are whatever a profile's control delimits (`frame`, from a header on transmit, a delimiter on
receive): the word goes before each, and is stripped from the start of each (Mode S: a preamble before frames of 56 or 112
bits). Soft values pass through; the decoded data's `errors` holds the word's mismatched bits per frame stripped.
"""
import numpy as np

from ..base import Mod, Digital, dtype
from ..base.rpps import Pipe, align
from ..base.types import Category, Rep
from ..sequence.word import word_bits
from ..base.control import anchor

class Marker(Pipe):
    CATEGORY = Category.CODE
    CONTROLS = {"frame": "live"} # A frame starts here (frames of no fixed length): the word goes before it, or is stripped
    _start = True

    def __init__(self, word, bits=None, length=None, order="msb"):
        self.word = word_bits(word, bits, order)
        self.n = len(self.word)
        self.length = None if length is None else int(length)
        if self.length is not None and self.length <= 0:
            raise ValueError("Marker: the frame length in bits after the word is positive (or None: frames by control)")

    def __str__(self):
        return f"Marker({self.n}+{self.length if self.length else 'frame'})"

    def control(self, frame=None):
        self._start = True

    def takes(self, encode):
        return {Rep.DATA, Rep.CHANNEL}

    def pos_rate(self, decode):
        if self.length is None:
            return 1.0 # Frames by control: one for one between the words, which are anchored
        L, n = self.length, self.n
        return L/(L + n) if decode else (L + n)/L

    def encode(self, data):
        ctx = dict(data.ctx)
        x = np.asarray(data.bits()).astype(bool)
        if self.length is None: # One frame per call (from a frame start): the word, then the bits (none yet: the word waits)
            if len(x) and self._start:
                self._start = False
                anchor(data.meta, self, data.meta.pos, self.n) # The frame's first bit follows the word
                x = np.concatenate((self.word, x))
            return data.derive(x, self.CATEGORY, dtype.BITS, Digital, ctx, rep=Rep.CHANNEL)
        x = align(ctx, (self, "tx"), x, self.length, pad=True)
        frames = x.reshape(-1, self.length)
        y = np.concatenate((np.broadcast_to(self.word, (len(frames), self.n)), frames), axis=1).reshape(-1)
        return data.derive(y, self.CATEGORY, dtype.BITS, Digital, ctx, rep=Rep.CHANNEL)

    def decode(self, data):
        ctx = dict(data.ctx)
        soft = isinstance(data, Mod) and data.is_soft
        x = np.asarray(data.data, dtype=np.float64) if soft else np.asarray(data.bits()).astype(bool)
        errors = None
        if self.length is None: # The word at each frame start is stripped (held while it is not all in yet)
            key = (self, "strip")
            if self._start:
                self._start = False
                ctx[key] = np.zeros(0, dtype=x.dtype)
            held = ctx.pop(key, None)
            y = x
            if held is not None:
                head = np.concatenate((held, x))
                if len(head) < self.n and not ctx.get("flush"):
                    ctx[key] = head
                    y = x[:0]
                else:
                    w = head[:self.n]
                    hard = w < 0 if soft else w.astype(bool)
                    errors = np.array([int(np.count_nonzero(hard != self.word[:len(hard)]))])
                    y = head[self.n:]
                    anchor(data.meta, self, data.meta.pos + self.n - len(held), 0) # The first bit after the word
        else:
            x = align(ctx, (self, "rx"), x, self.n + self.length, pad=False)
            frames = x.reshape(-1, self.n + self.length)
            head = frames[:, :self.n]
            hard = head < 0 if soft else head
            errors = np.count_nonzero(hard != self.word[None, :], axis=1)
            y = frames[:, self.n:].reshape(-1)
        if soft:
            out = data.derive(y.astype(dtype.LLR.dtype), self.CATEGORY, dtype.LLR, Mod, ctx, rep=Rep.DATA)
        else:
            out = data.derive(y, self.CATEGORY, dtype.BITS, Digital, ctx, rep=Rep.DATA)
        out.errors = errors
        return out
