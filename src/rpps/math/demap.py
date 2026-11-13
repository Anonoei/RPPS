"""Soft demapping of symbols to bit LLRs, for rows of symbols at once (one frame a row), on any array module `xp`

`rpps.symbol.Constellation.llr` is this on one row, and `rpps.parallel.kernels.Demap` on a batch of them: one implementation,
so a frame comes out the same alone or in a batch. Each row has its own noise level, estimated from its own symbols by EM over the
points (`mixture_variance` for the one-row case), and every sum over points keeps the nearest point's distance out of the
exponentials, so they stay in range.
"""
import numpy as np

from .. import _accel

def mask_matrix(labels):
    """(points, 2*bits) float matrix of the bit values of each point: the zeros first, then the ones. `labels`: (points, bits) bool"""
    labels = np.asarray(labels, dtype=bool)
    return np.concatenate((~labels, labels), axis=1).astype(np.float64)

def distances(xp, y, points):
    """Squared distances (rows, L, points) of the symbols (rows, L) from the points, and each symbol's nearest one"""
    pts = xp.asarray(points)
    y = y.astype(np.complex128)
    dr, di = y.real[:, :, None] - pts.real[None, None, :], y.imag[:, :, None] - pts.imag[None, None, :]
    d2 = dr*dr + di*di
    return d2, d2.min(axis=2)

def noise_variance(xp, d2, dmin, iters=8, floor=1e-12, tol=1e-10):
    """Variance per real dimension of each row (rows,), by EM of one shared variance over equiprobable points: `iters`
    iterations, fewer when every row has stopped moving (a relative change under `tol`; EM converges geometrically, in
    about four at the SNR a decoder works at)"""
    shifted = d2 - dmin[:, :, None]
    L = d2.shape[1]
    var = dmin.mean(axis=1)/2 + floor
    for _ in range(iters):
        r = xp.exp(-shifted/(2*var[:, None, None]))
        new = xp.maximum(((r*d2).sum(axis=2)/r.sum(axis=2)).sum(axis=1)/(2*L), floor)
        done = bool((xp.abs(new - var) <= tol*var).all())
        var = new
        if done:
            break
    return var

def llr_rows(xp, y, points, labels, masks, noise_var=None, maxlog=False, iters=8, floor=1e-6):
    """LLRs log(P(bit=0)/P(bit=1)), positive means 0, of symbols (rows, L): (rows, L*bits) float32, symbol after symbol.

    noise_var: complex noise variance E|n|^2, per row or one value; None: estimated from each row (floored at `floor`, per
    real dimension). labels: (points, bits) bool; masks: `mask_matrix(labels)`"""
    if xp is np and _accel.native is not None: # C++: a row at a time on several threads, only the exponentials that count
        nv = np.zeros(0) if noise_var is None else np.asarray(noise_var, dtype=np.float64).reshape(-1)
        return _accel.native.demap_llr(np.ascontiguousarray(y, dtype=np.complex128), np.asarray(points, dtype=np.complex128),
                                       np.asarray(labels, dtype=np.uint8), nv, bool(maxlog), int(iters), float(floor), 1e-10)
    return _llr_rows(xp, y, points, labels, masks, noise_var, maxlog, iters, floor)

def _llr_rows(xp, y, points, labels, masks, noise_var=None, maxlog=False, iters=8, floor=1e-6):
    """`llr_rows` in array operations (the reference for the C++ kernel, and what runs on another array module)"""
    n, L = y.shape
    bps = labels.shape[1]
    if L == 0:
        return xp.zeros((n, 0), dtype=np.float32)
    d2, dmin = distances(xp, y, points)
    if noise_var is None:
        nv = 2*noise_variance(xp, d2, dmin, iters, floor)
    else:
        nv = xp.broadcast_to(xp.asarray(noise_var, dtype=np.float64), (n,))
    nv = xp.maximum(nv, 2*floor)
    if maxlog:
        logp = -d2/nv[:, None, None]
        out = xp.empty((n, L, bps), dtype=np.float32)
        for j in range(bps):
            one = labels[:, j]
            out[:, :, j] = logp[:, :, ~one].max(axis=2) - logp[:, :, one].max(axis=2)
        return out.reshape(n, L*bps)
    # Exact: the sums over the points of each bit value are one product with the masks, of exponentials that the nearest
    # point's shift keeps in range (the shift cancels in the ratio). Where a sum underflows (a symbol far from every point of
    # a bit value) the log-sum-exp is done the long way, so the result is the same at any noise level
    ex = xp.exp(-(d2 - dmin[:, :, None])/nv[:, None, None])
    S = ex @ xp.asarray(masks)                                                # (rows, L, 2*bits): zeros then ones
    S0, S1 = S[:, :, :bps], S[:, :, bps:]
    tiny = 1e-250
    out = (xp.log(xp.maximum(S0, tiny)) - xp.log(xp.maximum(S1, tiny))).astype(np.float32)
    ii, jj, kk = xp.nonzero((S0 < tiny) | (S1 < tiny))
    if len(ii):
        lp = -d2[ii, jj, :]/nv[ii][:, None]                                    # (m, points)
        one = xp.asarray(labels[:, kk].T)                                      # (m, points): which points carry a 1
        out[ii, jj, kk] = (_lse_where(xp, lp, ~one) - _lse_where(xp, lp, one)).astype(np.float32)
    return out.reshape(n, L*bps)

def _lse_where(xp, lp, mask):
    """log(sum(exp(lp)) over the points where mask), without overflow"""
    x = xp.where(mask, lp, -xp.inf)
    m = x.max(axis=1, keepdims=True)
    m = xp.where(xp.isfinite(m), m, 0.0)
    return (m + xp.log(xp.exp(x - m).sum(axis=1, keepdims=True)))[:, 0]
