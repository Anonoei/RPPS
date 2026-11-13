"""Batch kernels of the generic operators (ARCHITECTURE.md, Parallelism)

Each is the batch form of one operator, built from its definition (never from a protocol): `Demap` of a constellation, and so
on. They take `[N, L]` arrays, one frame per row, and are checked row for row against the operator run frame by frame
(tests/unit/test_batch_kernels.py). They use only what numpy and cupy share, so `xp` can be either.
"""
import numpy as np

from ..base.data import Analog, Digital, Mod
from ..base.types import Category, Rep, dtype
from ..errors import RPPSError
from ..math.demap import llr_rows, mask_matrix
from .frame import Form
from .kernel import Kernel

_MIN_VAR = 1e-6 # the floor symbol.constellation uses

class Demap(Kernel):
    """Symbols to per-bit LLRs, as `Constellation.llr`: the noise level of each row (frame) estimated from its own symbols by EM
    over the points, then log(P(0)/P(1)) per bit (exact, or max-log). One row in gives `bits_per_symbol` LLRs per symbol"""
    CHUNK = 1 << 22 # (symbols x points) per step: bounds the memory of the distance table

    def __init__(self, points, labels, maxlog=False, name="demap", iters=8):
        self.points, self.labels, self.maxlog, self.iters = np.asarray(points, np.complex128), np.asarray(labels, bool), bool(maxlog), iters
        self.bps = self.labels.shape[1]
        self._masks = mask_matrix(self.labels)
        self.name = f"demap:{name}"

    def shape(self, n_in):
        return None if n_in is None else n_in*self.bps

    def consts(self):
        return {"points": self.points, "labels": self.labels}

    def run(self, xp, b):
        N, L = b.x.shape
        out = xp.zeros((N, L*self.bps), dtype=np.float32)
        if L:
            step = max(1, self.CHUNK//(L*len(self.points)))
            for a in range(0, N, step):
                out[a:a + step] = self._rows(xp, xp.asarray(b.x[a:a + step]))
        b.x = out
        b.form = Form(Mod, Category.SYMBOL, dtype.LLR, Rep.CHANNEL, *b.form[4:])
        return b

    def _rows(self, xp, y):
        return llr_rows(xp, y, self.points, self.labels, self._masks, maxlog=self.maxlog, iters=self.iters, floor=_MIN_VAR)

class BlockDecode(Kernel):
    """A decoder or deinterleaver of whole blocks, over a batch: the frames' blocks go through the coder in one call (a block code's
    decoder already runs its codewords on several threads, so a batch is more of them at once, not more calls). Each block is
    independent of the others, so a row comes out as it would alone. A frame is whole blocks; items past the last block are
    dropped and reported, as the operator does at the end of a flushed stream.

    coder: what the operator decodes with (`decode(bits or LLRs) -> bits or LLRs`, any number of blocks); block: items a
    block takes in; soft: it takes LLRs when it is given them"""
    def __init__(self, coder, block, soft, category, rep, name="block"):
        self.coder, self.block, self.soft, self.category, self.rep = coder, int(block), bool(soft), category, rep
        self.name = f"decode:{name}"

    def shape(self, n_in):
        return None # a code says how many items a block gives, not every coder how its input and output relate

    def _in(self, b, X):
        f = b.form
        if f.dt == dtype.LLR:
            return X if (self.soft and issubclass(f.cls, Mod)) else X < 0
        return X.astype(bool)

    def run(self, xp, b):
        N, L = b.x.shape
        used = L//self.block*self.block
        for i in range(N):
            if used < L and not b.bad[i]:
                b.notes.setdefault(i, []).append({"what": "cut short", "items": int(L - used), "at": int(b.pos[i] + b.n_in)})
        if used == 0:
            b.x = np.zeros((N, 0), dtype=bool)
            b.form = Form(Digital, self.category, dtype.BITS, self.rep, *b.form[4:])
            return b
        x = self._in(b, b.x[:, :used])
        try:
            y = np.asarray(self.coder.decode(x.reshape(-1))).reshape(N, -1)
        except RPPSError: # one frame's block was bad: find which, the others go on
            rows = []
            for i in range(N):
                try:
                    rows.append(np.asarray(self.coder.decode(x[i])).reshape(-1))
                except RPPSError as e:
                    b.fail(i, f"{type(e).__name__}: {e}")
                    rows.append(None)
            ok = next((r for r in rows if r is not None), None)
            if ok is None:
                b.x = np.zeros((N, 0), dtype=bool)
                return b
            y = np.zeros((N, len(ok)), dtype=ok.dtype)
            for i, r in enumerate(rows):
                if r is not None:
                    y[i] = r
        soft = y.dtype.kind == "f"
        b.x = y
        b.form = Form(Mod if soft else Digital, self.category, dtype.LLR if soft else dtype.BITS, self.rep, *b.form[4:])
        return b

_SYMBOLS = (Category.SYMBOL, dtype.SYMBOLS, Rep.SYMBOLS)

def _per_length(fn):
    """A function of the frame length, remembered (a plan sees few lengths; a constant is made once per length)"""
    cache = {}
    def get(L):
        v = cache.get(L)
        if v is None:
            v = cache[L] = fn(L)
        return v
    return get

class Rotate(Kernel):
    """Symbols turned by a fixed phasor each position: the batch of a phase scrambler, or any known rotation. `phasors(L)`: the
    unit phasors of a frame of L symbols, made once per length"""
    def __init__(self, phasors, name="rotate"):
        self.phasors = _per_length(phasors)
        self.name = f"rotate:{name}"

    def shape(self, n_in):
        return n_in

    def run(self, xp, b):
        L = b.x.shape[1]
        b.x = (b.x*xp.asarray(self.phasors(L))[None, :]).astype(np.complex64)
        b.form = Form(Mod, *_SYMBOLS, *b.form[4:])
        return b

class Gather(Kernel):
    """Symbols kept by a fixed selection of positions (pilots and headers removed): `keep(L)` is the boolean mask of a frame of L"""
    def __init__(self, keep, name="gather"):
        self.keep = _per_length(keep)
        self.name = f"gather:{name}"

    def shape(self, n_in):
        return None if n_in is None else int(self.keep(n_in).sum())

    def run(self, xp, b):
        b.x = xp.asarray(b.x)[:, xp.asarray(self.keep(b.x.shape[1]))].astype(np.complex64)
        b.form = Form(Mod, *_SYMBOLS, *b.form[4:])
        return b

class XorMask(Kernel):
    """Bits XOR a fixed mask repeated from the start of each frame (a scrambler restarted every `len(mask)` bits)"""
    def __init__(self, mask, category, rep, name="xor"):
        self.mask, self.category, self.rep = np.asarray(mask, dtype=bool), category, rep
        self.name = f"xor:{name}"

    def shape(self, n_in):
        return n_in

    def consts(self):
        return {"mask": self.mask}

    def run(self, xp, b):
        L = b.x.shape[1]
        x = b.x < 0 if b.form.dt == dtype.LLR else b.x.astype(bool)
        b.x = x ^ xp.asarray(self.mask[np.arange(L) % len(self.mask)])[None, :]
        b.form = Form(Digital, self.category, dtype.BITS, self.rep, *b.form[4:])
        return b

class PilotLine(Kernel):
    """The carrier phase of each frame from its pilot blocks, taken off: a least-squares line through the phase of every block (and
    through `lead`, a position where it is known to be 0), per row. Symbols past the frame pass unchanged, a frame cut short
    as it is. `starts`: where the blocks start in a frame, `total`: its length with the pilots"""
    def __init__(self, starts, total, length, value, lead, name="pilot_line"):
        self.starts, self.total, self.length, self.value, self.lead = np.asarray(starts, np.int64), int(total), int(length), complex(value), lead
        self.name = f"line:{name}"
        pos = [float(s) + self.length/2 for s in self.starts]
        self.pos = np.array(([float(lead)] if lead is not None else []) + pos)

    def shape(self, n_in):
        return n_in

    def consts(self):
        return {"starts": self.starts, "pos": self.pos}

    def run(self, xp, b):
        N, L = b.x.shape
        b.form = Form(Analog, Category.SYNC, dtype.SYMBOLS, Rep.SYMBOLS, *b.form[4:]) # (what a stage gives)
        if L < self.total or not len(self.pos):
            b.x = b.x.astype(np.complex64) # (a frame cut short is left as it came; with nothing to measure the symbols pass)
            return b
        Y = xp.asarray(b.x[:, :self.total]).astype(np.complex128)
        ph = xp.zeros((N, len(self.pos)))
        k = 1 if self.lead is not None else 0
        if len(self.starts):
            idx = xp.asarray(self.starts[:, None] + np.arange(self.length)[None, :])
            blocks = Y[:, idx]                                                   # (N, blocks, length)
            ph[:, k:] = xp.angle((blocks*np.conj(self.value)).mean(axis=2))
        ph = xp.unwrap(ph, axis=1)
        if len(self.pos) > 1:
            d = self.pos - self.pos.mean()
            slope = (ph*xp.asarray(d)[None, :]).sum(axis=1)/float((d*d).sum())
            icpt = ph.mean(axis=1) - slope*float(self.pos.mean())
        else:
            slope, icpt = xp.zeros(N), ph[:, 0]
        turn = icpt[:, None] + slope[:, None]*xp.arange(self.total)[None, :]
        out = xp.asarray(b.x).astype(np.complex128)
        out[:, :self.total] = Y*xp.exp(-1j*turn)
        b.x = out.astype(np.complex64)
        return b
