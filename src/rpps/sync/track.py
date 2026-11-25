"""Block-rate tracking (rpps.sync): a slow loop run once per window over the window estimates

A first-class estimator measures a parameter on each window (block estimation); this carries the parameter and its rate
from window to window, gating outliers, so a drifting clock or carrier is followed without a staircase: the value at
any position is `at(t)`, a line through the last update. It is a critically damped second-order loop: each window's
error (measured minus predicted, wrapped to the parameter's period) moves the value by `2*bw` of itself and the rate by
`bw**2` of it per window. `bw` is the loop's natural frequency in radians per window: smaller is quieter, larger follows
faster drift.

The state is a SimpleNamespace (`new()`), kept in the estimating stage's stream state, so a re-run can copy it.
"""
from types import SimpleNamespace

import numpy as np

def new():
    """A tracker that has seen nothing: `started` False until the first update"""
    return SimpleNamespace(started=False, value=0.0, rate=0.0, at=0.0, windows=0, missed=0)

def wrap(x, period):
    """x wrapped to [-period/2, period/2)"""
    return (x + period/2) % period - period/2

def at(tr, t):
    """The tracked value at position t (any position: the correction is evaluated per item, not per window)"""
    return tr.value + tr.rate*(t - tr.at)

def line_phasor(phi0, rate, length):
    """exp(-j (phi0 + rate i)) for i in 0 .. length-1: a phase line turned into a phasor with no exponential per item. Each
    block of about sqrt(length) items is one start phasor times the same short vector of exponentials, so it costs two
    multiplications an item and about 2 sqrt(length) exponentials, and agrees with the direct exponential to about 1e-14"""
    if length <= 0:
        return np.zeros(0, np.complex128)
    starts, local = line_parts(phi0, rate, length)
    return (starts[:, None]*local[None, :]).reshape(-1)[:length]

def line_parts(phi0, rate, length):
    """(starts, local) of `line_phasor`: item i's phasor is starts[i // len(local)]*local[i % len(local)] (a device kernel
    reads the line this way, the exponentials made on the host in double)"""
    block = max(16, int(max(length, 1)**0.5))
    nb = -(-max(length, 1)//block)
    return np.exp(-1j*(phi0 + rate*block*np.arange(nb))), np.exp(-1j*rate*np.arange(block))

def update(tr, measured, t, period=None, bw=0.25, rate=None):
    """A window's measurement (modulo `period` when given) at position t, its centre. rate: a measured rate to start from"""
    if not tr.started:
        tr.started, tr.value, tr.at = True, float(measured), float(t)
        tr.rate = 0.0 if rate is None else float(rate)
        tr.windows, tr.missed = 1, 0
        return 0.0
    dt = t - tr.at
    pred = at(tr, t)
    e = measured - pred
    if period is not None:
        e = wrap(e, period)
    tr.value, tr.at = pred + 2*bw*e, float(t)
    if dt > 0:
        tr.rate += bw*bw*e/dt
    tr.windows += 1
    tr.missed = 0
    return float(e)

def miss(tr, t):
    """A window gated out: the tracker coasts on its rate"""
    if tr.started:
        tr.value, tr.at = at(tr, t), float(t)
        tr.missed += 1

# ---- Windows at absolute stream positions (rpps.sync, feedforward rules)

def windows(st, x, meta, W):
    """Hold x and give each window ready: (a0, b, seg) for stream positions [a0, b) of [k*W, (k+1)*W). On a flush the
    last, partial window comes too. `st` carries `buf` (the items held from position `base`), `base`, `k` and `W`:
    None until the first block sets them. Afterwards call `keep(st, pos)` to let go of what lies before pos"""
    if st.base is None:
        st.base, st.W, st.k = meta.pos, W, meta.pos//W
    st.buf = np.concatenate((st.buf, np.asarray(x, dtype=st.buf.dtype)))
    end = st.base + len(st.buf)
    while True:
        a, b = st.k*st.W, (st.k + 1)*st.W
        if b > end:
            if not (meta.flush and a < end):
                return
            b = end
        a0 = max(a, st.base)
        yield a0, b, st.buf[a0 - st.base:b - st.base]
        st.k += 1
        if b >= end:
            return

def keep(st, pos):
    """Let go of the items held before stream position pos (at most all of them)"""
    pos = min(max(pos, st.base), st.base + len(st.buf))
    st.buf, st.base = st.buf[pos - st.base:], pos
