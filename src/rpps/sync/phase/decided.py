"""Carrier phase from decisions, by block estimation tracked across windows: any constellation (APSK, QAM, PSK)

Symbols are cut into windows of `window` at absolute positions. Each window is turned back by the phase the tracker
(rpps.sync.track) predicts for it, and its residual is the angle of sum y conj(d), d the nearest point to y (the
window's amplitude matched to the points' first), twice: the second time decided on symbols already closer to their
points. The tracker carries phase and frequency to the next window, so the correction is a line per symbol and a
frequency offset is followed. No loop over symbols: one per window.

Decisions are right only near the points: the phase has to start within half the points' closest angular spacing
(16APSK's outer ring: 15 degrees), so this follows a carrier a preamble or a word has taken off, as a Costas loop on
the same points would; it does not acquire one (`MPower` does, for M-fold symbols). `follow(y, points)` corrects a
whole array; `DDCarrier` is the stream stage.
"""
from types import SimpleNamespace

import numpy as np

from ...base import Stage
from .. import track

CHUNK = 1 << 16 # Symbols decided at a time (the distance matrix is symbols x points)

def decide(y, points):
    """The nearest of `points` to each of y"""
    pts = np.asarray(points, dtype=np.complex128).ravel()
    flat = np.asarray(y).reshape(-1)
    out = np.empty(len(flat), np.complex128)
    for a in range(0, len(flat), CHUNK):
        u = flat[a:a + CHUNK]
        out[a:a + CHUNK] = pts[np.argmin(np.abs(u[:, None] - pts[None, :]), axis=1)]
    return out.reshape(np.shape(y))

def residual(y, points, iters=2):
    """The phase of y against its own decisions (its amplitude matched to the points'), refined `iters` times"""
    pts = np.asarray(points, dtype=np.complex128).ravel()
    scale = np.sqrt(np.mean(np.abs(pts)**2)/max(float(np.mean(np.abs(y)**2)), 1e-30))
    e = 0.0
    for _ in range(iters):
        u = y*np.exp(-1j*e)
        e += float(np.angle(np.sum(u*np.conj(decide(u*scale, pts)))))
    return e

class DDCarrier(Stage):
    """Carrier phase and frequency from decisions on any constellation, block estimation tracked across windows

        symbols + rp.sync.phase.DDCarrier(rp.symbol.load("apsk/dvbs2_16apsk", rate="2/3").points)

    First class (rpps.sync); see the module. Symbols in (one a symbol), the same out. `window` symbols a window, `bw` the
    tracker's (rpps.sync.track), wide for the first windows while it pulls in. 32 symbols: on 16APSK with a
    drifting, noisy phase, 64 slipped a rotation once in 40 runs while the rate was being learned, 32 never. Latency: a window. Estimates in
    stream.est[name]: phase (rad), frequency (cycles a symbol), latency (symbols).
    """
    CONTROLS = {}

    def __init__(self, points, window=32, bw=0.25, name="carrier"):
        self.points = np.asarray(points, dtype=np.complex128).ravel()
        self.window, self.bw, self.name = int(window), bw, name
        if self.window < 2:
            raise ValueError("DDCarrier: a window of 2 symbols at least")

    def __str__(self):
        return "DDCarrier"

    def pos_rate(self, decode):
        return 1.0

    def new_state(self, meta):
        return SimpleNamespace(buf=np.zeros(0, np.complex128), base=None, k=None, W=None, tr=track.new())

    def step(self, x, st, meta):
        out = []
        for a0, b, seg in track.windows(st, x, meta, self.window):
            tr, c = st.tr, (a0 + b - 1)/2
            if len(seg) and np.any(seg):
                n = np.arange(a0, b)
                pred = track.at(tr, n) if tr.started else np.zeros(len(n))
                r = residual(seg*np.exp(-1j*pred), self.points)
                at_c = track.at(tr, c) if tr.started else 0.0
                track.update(tr, at_c + r, c, bw=self.bw if tr.windows >= 2 else max(self.bw, 0.5))
            out.append(seg*np.exp(-1j*track.at(tr, np.arange(a0, b))) if tr.started else seg.astype(np.complex128))
        track.keep(st, st.k*st.W)
        tr = st.tr
        meta.stream.est[self.name] = dict(phase=float(tr.value) if tr.started else None,
                                          frequency=tr.rate/(2*np.pi) if tr.started else None, latency=self.window)
        return np.concatenate(out).astype(np.complex64) if out else np.zeros(0, np.complex64)

def follow(y, points, window=32, bw=0.25):
    """y (symbols, one each, a carrier near 0 already) with the phase its decisions show taken off: `DDCarrier` over the
    whole array"""
    from ...base.data import Data
    y = np.asarray(y)
    if len(y) == 0:
        return y.astype(np.complex128)
    return np.asarray((Data(y.astype(np.complex64), sps=1, flush=True) + DDCarrier(points, window, bw)).data).astype(np.complex128)
