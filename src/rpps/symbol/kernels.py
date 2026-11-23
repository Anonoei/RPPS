"""Batch kernels of the symbol operators: `Demap` of a constellation, `Rotate` of a phase scrambler, `Gather` of a pilot or
header remover

Checked row for row against the operator run frame by frame (tests/unit/test_batch_kernels.py); they use only what
numpy and cupy share, so `xp` can be either (rpps.parallel). `Demap` on a GPU is a device kernel of its own (DEMAP_SOURCE)
"""
import numpy as np

from ..base.data import Mod
from ..base.kernel import Form, Kernel, device_functions, has_device_kernels
from ..base.types import Category, Rep, dtype
from ..math.demap import llr_rows, mask_matrix


# The demapper on a device, as `rpps.math.demap.llr_rows` and the C++ demap_llr: a thread a symbol. demap_em gives each
# symbol's EM term (sum r d2 / sum r, r = exp(-(d2 - dmin)/(2 var))) and its nearest distance, for the row's variance;
# demap_llr each bit's log(P0/P1), every log-sum-exp shifted by its own maximum (exact at any noise level, nothing
# underflows) or max-log. float32, distances made again per bit rather than kept (no local arrays). Portable CUDA C.
DEMAP_SOURCE = r"""
#define DEMAP_INF 3.402823466e+38f
extern "C" __global__ void demap_em(const float *y, const float *px, const float *py, const int P, const int L, const int n,
                                    const float *inv2var, float *term, float *dmin) {
    const long long t = (long long)blockIdx.x*blockDim.x + threadIdx.x;
    if (t >= (long long)n*L) return;
    const float yr = y[2*t], yi = y[2*t + 1], iv = inv2var[t/L];
    float m = DEMAP_INF;
    for (int p = 0; p < P; p++) { const float dr = yr - px[p], di = yi - py[p]; m = fminf(m, dr*dr + di*di); }
    float sr = 0.0f, srd = 0.0f;
    for (int p = 0; p < P; p++) {
        const float dr = yr - px[p], di = yi - py[p], d2 = dr*dr + di*di, r = expf(-(d2 - m)*iv);
        sr += r; srd += r*d2;
    }
    term[t] = srd/sr;
    dmin[t] = m;
}
extern "C" __global__ void demap_llr(const float *y, const float *px, const float *py, const int *lab, const int P,
                                     const int bps, const int L, const int n, const float *inv_nv, const int maxlog,
                                     float *out) {
    const long long t = (long long)blockIdx.x*blockDim.x + threadIdx.x;
    if (t >= (long long)n*L) return;
    const float yr = y[2*t], yi = y[2*t + 1], iv = inv_nv[t/L];
    for (int j = 0; j < bps; j++) {
        const int bit = 1 << (bps - 1 - j);
        float m0 = -DEMAP_INF, m1 = -DEMAP_INF;
        for (int p = 0; p < P; p++) {
            const float dr = yr - px[p], di = yi - py[p], lp = -(dr*dr + di*di)*iv;
            if (lab[p] & bit) m1 = fmaxf(m1, lp); else m0 = fmaxf(m0, lp);
        }
        float v = m0 - m1;
        if (!maxlog) {
            float s0 = 0.0f, s1 = 0.0f;
            for (int p = 0; p < P; p++) {
                const float dr = yr - px[p], di = yi - py[p], lp = -(dr*dr + di*di)*iv;
                if (lab[p] & bit) s1 += expf(lp - m1); else s0 += expf(lp - m0);
            }
            v += logf(s0) - logf(s1);
        }
        out[t*bps + j] = v;
    }
}
"""

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

    THREADS = 256

    def run(self, xp, b):
        N, L = b.x.shape
        if L and has_device_kernels(xp):
            out = self._device(xp, b.x)
        else:
            out = xp.zeros((N, L*self.bps), dtype=np.float32)
            if L:
                step = max(1, self.CHUNK//(L*len(self.points)))
                for a in range(0, N, step):
                    out[a:a + step] = self._rows(xp, b.x[a:a + step])
        b.x = out
        b.form = Form(Mod, Category.SYMBOL, dtype.LLR, Rep.CHANNEL, *b.form[4:])
        return b

    def _device(self, xp, y):
        """LLRs of the rows on the device: the noise of each row by `iters` EM steps (all of them: a converged step moves
        less than the host's stopping tolerance), then the bits"""
        n, L = y.shape
        f = device_functions(xp, DEMAP_SOURCE, ("demap_em", "demap_llr"))
        px = self.const(xp, "px", lambda: self.points.real.astype(np.float32))
        py = self.const(xp, "py", lambda: self.points.imag.astype(np.float32))
        bits = 1 << np.arange(self.bps - 1, -1, -1)
        lab = self.const(xp, "labmask", lambda: (self.labels.astype(np.int64) @ bits).astype(np.int32))
        P = np.int32(len(self.points))
        y = xp.ascontiguousarray(y, dtype=np.complex64)
        T = self.THREADS
        grid = ((n*L + T - 1)//T,)
        term = xp.empty((n, L), dtype=np.float32)
        dmin = xp.empty((n, L), dtype=np.float32)
        args = lambda iv: (y, px, py, P, np.int32(L), np.int32(n), iv, term, dmin)
        f["demap_em"](grid, (T,), args(xp.zeros(n, dtype=np.float32)))
        var = dmin.sum(axis=1, dtype=np.float64)/L/2 + self.floor
        for _ in range(self.iters):
            f["demap_em"](grid, (T,), args((0.5/var).astype(np.float32)))
            var = xp.maximum(term.sum(axis=1, dtype=np.float64)/(2*L), self.floor)
        nv = xp.maximum(2*var, 2*self.floor)
        out = xp.empty((n, L*self.bps), dtype=np.float32)
        f["demap_llr"](grid, (T,), (y, px, py, lab, P, np.int32(self.bps), np.int32(L), np.int32(n), (1.0/nv).astype(np.float32),
                                    np.int32(self.maxlog), out))
        return out

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
        # (an integer index, made once: a boolean one costs a device a search and a wait on every batch)
        b.x = b.x[:, self.const(xp, L, lambda: np.nonzero(self.keep(L))[0])].astype(np.complex64)
        b.form = Form(Mod, *_SYMBOLS, *b.form[4:])
        return b
