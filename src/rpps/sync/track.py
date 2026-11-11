"""Block-rate tracking (ARCHITECTURE.md, Sync methods): a slow loop run once per window over the window estimates

A first-class estimator measures a parameter on each window (block estimation); this carries the parameter and its rate
from window to window, gating outliers, so a drifting clock or carrier is followed without a staircase: the value at
any position is `at(t)`, a line through the last update. It is a critically damped second-order loop: each window's
error (measured minus predicted, wrapped to the parameter's period) moves the value by `2*bw` of itself and the rate by
`bw**2` of it per window. `bw` is the loop's natural frequency in radians per window: smaller is quieter, larger follows
faster drift.

The state is a SimpleNamespace (`new()`), kept in the estimating stage's stream state, so a re-run can copy it.
"""
from types import SimpleNamespace


def new():
    """A tracker that has seen nothing: `started` False until the first update"""
    return SimpleNamespace(started=False, value=0.0, rate=0.0, at=0.0, windows=0, missed=0)

def wrap(x, period):
    """x wrapped to [-period/2, period/2)"""
    return (x + period/2) % period - period/2

def at(tr, t):
    """The tracked value at position t (any position: the correction is evaluated per item, not per window)"""
    return tr.value + tr.rate*(t - tr.at)

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
