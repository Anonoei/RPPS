"""One-shot rational resampling of an array"""
from math import gcd

import numpy as np

from ..filters.impl.designer import designer

_CHUNK = 1 << 22 # Elements of the window matrix built at a time

def design(up, down, window=("kaiser", 5.0)):
    """The filter `resample_poly` uses: (taps at the up-sampled rate, gain `up` included, centre index)"""
    n = max(up, down)
    c = 10*n
    if isinstance(window, tuple) and window and window[0] == "kaiser":
        beta = float(window[1]) if len(window) > 1 else 5.0
        win = lambda m: np.kaiser(m, beta)
    elif callable(window):
        win = window
    elif isinstance(window, tuple):
        raise ValueError(f"unknown window {window!r}: use ('kaiser', beta), a function of the length, or the taps themselves")
    else:
        h = np.asarray(window, np.float64)
        return h, (len(h) - 1)//2
    return up*designer(2*c + 1, 1/n, window=win, scale=True), c

def resample_poly(x, up, down, window=("kaiser", 5.0)):
    """Resample `x` by the rational factor up/down with a linear-phase FIR low pass and no delay: output m is the input at time m*down/up

    Only the outputs that are kept are computed (polyphase). Returns ceil(len(x)*up/down) samples; single precision stays single.
    window: ("kaiser", beta), a function of the length, or the taps themselves (at the up-sampled rate, odd length)
    """
    x = np.asarray(x)
    up, down = int(up), int(down)
    if up < 1 or down < 1:
        raise ValueError("up and down must be positive whole numbers")
    g = gcd(up, down)
    up, down = up//g, down//g
    if x.ndim != 1:
        raise ValueError("resample_poly takes a 1-D array")
    single = x.dtype in (np.float32, np.complex64)
    if up == down == 1:
        return x.copy()
    h, c = design(up, down, window)
    nout = -(-len(x)*up//down)
    cplx = np.iscomplexobj(x)
    work = np.complex64 if (single and cplx) else np.float32 if single else np.complex128 if cplx else np.float64
    h = h.astype(np.float32 if single else np.float64, copy=False)
    out = np.empty(nout, dtype=work)
    if nout == 0:
        return out
    J = (c + up - 1)//up + 1                 # taps on each side of an input sample at the up-sampled phase
    K = 2*J + 1
    # y[m] = sum_j x[i0 - j] h[ph + c + j up], i0 = m down // up, ph = m down % up; every up-th output shares ph
    xp = np.concatenate((np.zeros(J, x.dtype), x, np.zeros(J + down + 1, x.dtype))).astype(work, copy=False)
    win = np.lib.stride_tricks.sliding_window_view(xp, K)
    for u in range(min(up, nout)):
        ph = (u*down) % up
        j = np.arange(J, -J - 1, -1)          # j = J - q for the window element q
        idx = ph + c + j*up
        gr = np.where((idx >= 0) & (idx <= 2*c), h[np.clip(idx, 0, len(h) - 1)], 0).astype(h.dtype)
        i0 = (u*down)//up
        nt = -(-(nout - u)//up)
        rows = win[i0:i0 + nt*down:down][:nt]
        step = max(_CHUNK//K, 1)
        for a in range(0, nt, step):
            blk = rows[a:a + step]
            if cplx:
                y = (np.ascontiguousarray(blk.real) @ gr) + 1j*(np.ascontiguousarray(blk.imag) @ gr)
            else:
                y = blk @ gr
            out[u + up*a:u + up*(a + len(blk)):up] = y
    return out
