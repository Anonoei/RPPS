"""Batch kernels of the symbol operators: `Demap` of a constellation, `Rotate` of a phase scrambler, `Gather` of a pilot or
header remover

Checked row for row against the operator run frame by frame (tests/unit/test_batch_kernels.py); they use only what
numpy and cupy share, so `xp` can be either (rpps.parallel)
"""
import numpy as np

from ..base.data import Mod
from ..base.kernel import Form, Kernel
from ..base.types import Category, Rep, dtype
from ..math.demap import llr_rows, mask_matrix


class Demap(Kernel):
    """Symbols to per-bit LLRs, as `Constellation.llr`: the noise level of each row (frame) estimated from its own symbols by EM
    over the points, then log(P(0)/P(1)) per bit (exact, or max-log). One row in gives `bits_per_symbol` LLRs per symbol"""
    CHUNK = 1 << 22 # (symbols x points) per step: bounds the memory of the distance table

    def __init__(self, points, labels, maxlog=False, name="demap", iters=8, floor=1e-6):
        self.floor = floor
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
                out[a:a + step] = self._rows(xp, b.x[a:a + step])
        b.x = out
        b.form = Form(Mod, Category.SYMBOL, dtype.LLR, Rep.CHANNEL, *b.form[4:])
        return b

    def _rows(self, xp, y):
        pts, lab, masks = (self.const(xp, "points", lambda: self.points), self.const(xp, "labels", lambda: self.labels),
                           self.const(xp, "masks", lambda: self._masks)) # (on a device once)
        return llr_rows(xp, y, pts, lab, masks, maxlog=self.maxlog, iters=self.iters, floor=self.floor)

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
        b.x = (b.x*self.const(xp, L, lambda: self.phasors(L))[None, :]).astype(np.complex64)
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
        L = b.x.shape[1]
        b.x = b.x[:, self.const(xp, L, lambda: self.keep(L))].astype(np.complex64)
        b.form = Form(Mod, *_SYMBOLS, *b.form[4:])
        return b
