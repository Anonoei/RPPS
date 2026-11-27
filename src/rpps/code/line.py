"""Line codes: bits <-> bits, one step of the code transform, by name

    bits * rp.code.Line("nrzi")              # AX.25: a 0 is a transition, a 1 no change
    bits * rp.code.Line("nrzm")              # differential bits: a 1 is a transition
    bits * rp.code.Line("manchester")        # 0 -> 10, 1 -> 01 (IEEE 802.3); "thomas": 0 -> 01, 1 -> 10
    bits * rp.code.Line("invert")            # every bit inverted (data inversion)
    bits * rp.code.Line("biphase_m")         # IRIG-106 Bi-phi-M: a transition at every bit start, and mid-bit for a 1
    bits * rp.code.Line("biphase_s")         # Bi-phi-S: the same, mid-bit for a 0
    bits * rp.code.Line("miller")            # delay modulation (DM-M): mid-bit for a 1, at the boundary between two 0s

Bi-phase-M/S and Miller are 2 channel bits a data bit and decode from the two halves alone (a mid-bit transition or not),
so they need no reference level and ignore an inverted signal.

Decoding keeps soft values soft: an inversion negates LLRs, a differential code takes the box-plus of neighbours
(sign times the smaller magnitude), Manchester the difference of the two halves. In a profile: {"code": "line", "name": "nrzi"}.
Manchester as a pulse (first half +1, second -1, at the sample level) is `rp.signal.filters.ShapingBiphase` instead
(rpps.base, Grey areas): this is the bit-level form, 2 channel bits per data bit.
"""
import numpy as np

from ..base import Mod, Digital, dtype
from ..base.rpps import Pipe, align
from ..base.types import Category, Rep
from ..base.control import anchor

NAMES = ("invert", "nrzi", "nrzm", "manchester", "thomas", "biphase_m", "biphase_s", "miller")
MID = {"biphase_m": True, "biphase_s": False, "miller": True} # the bit a mid-bit transition sends

class Line(Pipe):
    CATEGORY = Category.CODE
    CONTROLS = {"frame": "live"} # A frame starts here: Manchester pairs align to it, a differential code starts from a 0
    _restart = False

    def __init__(self, name="nrzi"):
        if name not in NAMES:
            raise ValueError(f"line code is one of {NAMES}, not {name!r}")
        self.name = name
        self.pairs = name in ("manchester", "thomas") or name in MID

    def control(self, frame=None):
        self._restart = True

    def _frame_start(self, data, ctx, key, out_index=0):
        """At a frame start (a profile's control) forget what the last frame left, and say where this one's output begins"""
        if self._restart:
            self._restart = False
            ctx.pop(key, None)
            anchor(data.meta, self, data.meta.pos, out_index)

    def __str__(self):
        return f"Line({self.name})"

    def takes(self, encode):
        return {Rep.DATA, Rep.CHANNEL}

    def pos_rate(self, decode):
        if not self.pairs:
            return 1.0
        return 0.5 if decode else 2.0

    CONFIDENT = 0.5 # Soft halves both this far (times the block's mean magnitude) from 0 and of one sign: an invalid pair

    # ---- encode

    def encode(self, data):
        ctx = dict(data.ctx)
        x = np.asarray(data.bits()).astype(bool)
        key = (self, "tx")
        self._frame_start(data, ctx, key)
        if self.name == "invert":
            y = ~x
        elif self.name in ("nrzi", "nrzm"):
            # y[n] = y[n-1] ^ x[n] (nrzm), y[n] = y[n-1] ^ ~x[n] (nrzi): the running parity of the transitions
            t = x if self.name == "nrzm" else ~x
            last = ctx.get(key, False)
            y = (np.cumsum(t.astype(np.int64)) + int(last)) % 2 == 1
            if len(y):
                ctx[key] = bool(y[-1])
        elif self.name in MID:
            level, prev = ctx.get(key, (False, True)) # the line level, the bit before (Miller: a 1, so no boundary transition)
            # Each half bit's level is the level before XOR the parity of the transitions up to it: one at the start of a
            # bit (Bi-phase always; Miller between two 0s), one mid-bit where the bit is the one MID sends
            flips = np.empty(2*len(x), dtype=bool)
            if self.name == "miller":
                before = np.concatenate(([prev], x[:-1])) if len(x) else x
                flips[0::2] = ~x & ~before
            else:
                flips[0::2] = True
            flips[1::2] = x == MID[self.name]
            y = (np.cumsum(flips, dtype=np.int64) % 2 == 1) ^ bool(level)
            if len(x):
                level, prev = bool(y[-1]), bool(x[-1])
            ctx[key] = (level, prev)
        else:
            y = np.empty(2*len(x), dtype=bool)
            first = ~x if self.name == "manchester" else x # IEEE: 0 -> 10; Thomas: 0 -> 01
            y[0::2], y[1::2] = first, ~first
        return data.derive(y, self.CATEGORY, dtype.BITS, Digital, ctx, rep=Rep.CHANNEL)

    # ---- decode

    def decode(self, data):
        ctx = dict(data.ctx)
        soft = isinstance(data, Mod) and data.is_soft
        x = np.asarray(data.data, dtype=np.float64) if soft else np.asarray(data.bits()).astype(bool)
        key = (self, "rx")
        self._frame_start(data, ctx, key)
        if self.name == "invert":
            y = -x if soft else ~x
        elif self.name in ("nrzi", "nrzm"):
            prev = ctx.get(key)
            if prev is None:
                prev = 50.0 if soft else False # Before the stream: a 0, held sure (the first bit decodes against it)
            p = np.concatenate(([prev], x[:-1])) if len(x) else x
            if len(x):
                ctx[key] = x[-1]
            if soft:
                y = np.sign(x)*np.sign(p)*np.minimum(np.abs(x), np.abs(p)) # box-plus of neighbours: the LLR of their XOR
                if self.name == "nrzi":
                    y = -y
            else:
                y = x ^ p
                if self.name == "nrzi":
                    y = ~y
        elif self.name in MID: # the bit is whether the halves differ (MID) or agree
            x = align(ctx, key, x, 2, pad=False)
            a, b = x[0::2], x[1::2]
            if soft: # box-plus: the LLR that the halves are equal
                y = np.sign(a)*np.sign(b)*np.minimum(np.abs(a), np.abs(b))
                y = y if MID[self.name] else -y # (equal halves: a 0 where a mid-bit transition sends a 1)
            else:
                y = (a != b) if MID[self.name] else (a == b)
            bad = np.zeros(len(y), bool) # (any pair of halves is a valid bit)
        else:
            x = align(ctx, key, x, 2, pad=False)
            a, b = x[0::2], x[1::2]
            if soft: # LLR of the bit from its two halves: positive (a 0) when the halves are as a 0 sends them
                y = 0.5*(b - a) if self.name == "manchester" else 0.5*(a - b)
                sure = self.CONFIDENT*np.mean(np.abs(x)) if len(x) else 0.0
                bad = (np.sign(a) == np.sign(b)) & (np.minimum(np.abs(a), np.abs(b)) > sure)
            else:
                y = b if self.name == "manchester" else a
                bad = a == b # Manchester: an invalid pair (00, 11) is marked `failed` (a link header drops its frame)
        if soft:
            out = data.derive(y.astype(dtype.LLR.dtype), self.CATEGORY, dtype.LLR, Mod, ctx, rep=Rep.DATA)
        else:
            out = data.derive(y, self.CATEGORY, dtype.BITS, Digital, ctx, rep=Rep.DATA)
        if self.pairs:
            out.failed = bad if bad.any() else None
        return out
