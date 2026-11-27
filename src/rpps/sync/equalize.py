"""Equalizers: `symbols + rp.sync.equalize.BlockLS(qpsk)` solves the taps per window

The taps carry across blocks, so any block length gives the same stream. The per-symbol adaptive equalizers (second
class) are `rp.sync.loops.LMS` and `rp.sync.loops.CMA`.
"""
from types import SimpleNamespace

import numpy as np

from ..base import Stage

class BlockLS(Stage):
    """Equalize symbols by least squares over each window, decision directed, with the taps carried from window to window

        symbols = timed + carrier + rp.sync.equalize.BlockLS(rp.symbol.load("QPSK"), taps=7)

    First class (rpps.sync): block estimation, where `loops.LMS` and `loops.CMA` adapt per symbol. Windows sit at
    absolute stream positions, `window` seconds (at least `min_symbols`). On each, the taps w (symbol spaced, centred:
    output n uses inputs n - taps//2 .. n + taps//2) solve min |X w - d|^2, d the constellation points nearest to X w,
    `iters` times; the window's solution is blended with the last (`blend` of the old: the tracking across windows), and
    that is the filter for the window. It starts as a pass-through and normalizes the power to the constellation's.
    A window that is noise only would pull the taps anywhere: `gate` is the least fraction of decisions within half the
    minimum distance for a window to update the taps. Latency: a window and taps//2 symbols (est: `latency`).

    `blind=True`: a window whose decisions fail the gate (an eye the channel has closed, before any decision is right) is
    solved without decisions, by block least squares constant modulus (LS-CMA, Agee): `blind_iters` times, the targets
    are the outputs y pushed onto the constellation's modulus (R y/|y|, R^2 = E|a|^4/E|a|^2), and w solves
    min |X w - R y/|y||^2 with the same normal equations. The modulus does not see the phase: the taps are turned by the
    constellation's M-th power phase (M its rotational symmetry: 2 BPSK, 4 QPSK/QAM, 8 8PSK) and, of the M rotations left,
    the one nearest the taps before, so windows blend alike. The window is then refined decision directed (when its
    decisions pass the gate within dmin/4, not dmin/2: a dense constellation's own spread lands within dmin/2 of a point
    pi/4 of the time, eye open or not); the M-fold ambiguity is left to what follows (a frame word, differential coding).
    est: `mode`, "decided" or "blind" (the last window's).
    """
    CONTROLS = {}

    def __init__(self, constellation, taps=7, window=1e-3, min_symbols=1024, iters=3, blend=0.5, gate=0.5, name="equalizer",
                 blind=False, blind_iters=8):
        if taps < 1 or taps % 2 == 0:
            raise ValueError("taps is an odd number")
        self.points = np.asarray(constellation.points, dtype=np.complex128)
        self.dmin = float(constellation.min_distance)
        self.taps, self.window, self.min_symbols = int(taps), window, min_symbols
        self.iters, self.blend, self.gate, self.name = int(iters), float(blend), float(gate), name
        self.blind, self.blind_iters = bool(blind), int(blind_iters)
        a2 = np.mean(np.abs(self.points)**2)
        self.modulus = float(np.sqrt(np.mean(np.abs(self.points)**4)/a2))
        self.order, self.moment = symmetry(self.points)

    def __str__(self):
        return f"BlockLS({self.taps})"

    def pos_rate(self, decode):
        return 1.0

    def new_state(self, meta):
        h = self.taps//2
        w = np.zeros(self.taps, np.complex128)
        w[h] = 1.0
        return SimpleNamespace(buf=np.zeros(h, np.complex128), base=meta.pos - h, k=None, W=None, w=w, scaled=False,
                               good=0.0, windows=0, mode=None)

    def _decide(self, y):
        return self.points[np.argmin(np.abs(y[:, None] - self.points[None, :]), axis=1)]

    @staticmethod
    def _solve(X, R, XH, d):
        try:
            return np.linalg.solve(R, XH @ d)
        except np.linalg.LinAlgError: # (a window of zeros)
            return np.linalg.lstsq(X, d, rcond=None)[0]

    def _decided(self, X, XH, R, w, radius):
        """Decision directed least squares, `iters` times from w: (the taps, the fraction of decisions within `radius`),
        the taps None when that fails the gate"""
        good = 0.0
        for _ in range(self.iters):
            y = X @ w
            d = self._decide(y)
            good = float(np.mean(np.abs(y - d) < radius))
            if good < self.gate:
                return None, good
            w = self._solve(X, R, XH, d)
        return w, good

    def _cma(self, X, XH, R, w):
        """Least squares constant modulus from w, `blind_iters` times, turned to the constellation's phase and to the
        rotation (of its symmetry) nearest w"""
        w0 = w
        for _ in range(self.blind_iters):
            y = X @ w
            w = self._solve(X, R, XH, self.modulus*y/np.maximum(np.abs(y), 1e-30))
        y = X @ w
        if self.moment != 0: # the M-th power phase: E[y^M] = E[a^M] e^{jM phi}
            w = w*np.exp(-1j*np.angle(np.mean(y**self.order)/self.moment)/self.order)
        return self._align(w, w0)

    def _align(self, w, ref):
        """w turned by the one of its symmetry's rotations that brings it nearest `ref`"""
        if self.order < 1:
            return w
        turns = np.exp(2j*np.pi*np.arange(self.order)/self.order)
        return w*turns[np.argmax(np.real(turns*np.vdot(ref, w)))]

    def step(self, x, st, meta):
        h = self.taps//2
        if st.W is None:
            st.W = max(int(round(self.window*meta.Fs)), int(self.min_symbols))
            st.k = meta.pos//st.W
        st.buf = np.concatenate((st.buf, np.asarray(x, dtype=np.complex128)))
        end = st.base + len(st.buf) # input positions held: [base, end)
        real = end
        W, out = st.W, []
        while True:
            a, b = st.k*W, (st.k + 1)*W
            need = b + h
            if need > end:
                if not meta.flush or a >= end - h:
                    break
                pad = need - end
                st.buf = np.concatenate((st.buf, np.zeros(pad)))
                end = need
            a0 = max(a, st.base + h)
            X = np.lib.stride_tricks.sliding_window_view(st.buf[a0 - h - st.base:b + h - st.base], self.taps)[:, ::-1]
            if not st.scaled: # Unit power, as the constellation
                st.w = st.w/np.sqrt(max(np.mean(np.abs(X[:, h])**2), 1e-30))
                st.scaled = True
            XH = X.conj().T
            R = XH @ X # The normal equations' matrix: the same for every iteration of the window, only the targets change
            # (blind: decisions only once the eye is open, within dmin/4: within dmin/2 a dense constellation's own
            # spread passes, pi/4 of every square cell)
            radius = self.dmin/4 if self.blind else self.dmin/2
            w, good = self._decided(X, XH, R, st.w, radius)
            st.mode = "decided"
            if w is None and self.blind and np.any(X[:, h]):
                w = self._cma(X, XH, R, st.w)
                st.mode = "blind"
                refined, good = self._decided(X, XH, R, w, radius)
                if refined is not None:
                    w = self._align(refined, st.w)
                else:
                    good = float(np.mean(np.abs(X @ w - self._decide(X @ w)) < radius))
            if w is not None:
                st.w = self.blend*st.w + (1 - self.blend)*w
                st.windows += 1
            st.good = good
            out.append((X @ st.w)[:max(0, min(b, real) - a0)]) # (past the stream's end on a flush: padding)
            st.k += 1
            if meta.flush and b + h >= end:
                break
        keep = min(max(st.k*W - h, st.base), end)
        st.buf, st.base = st.buf[keep - st.base:], keep
        meta.stream.est[self.name] = dict(taps=st.w.copy(), good=st.good, windows=st.windows, mode=st.mode,
                                          latency=(W + h)/meta.Fs)
        return np.concatenate(out).astype(np.complex64) if out else np.zeros(0, np.complex64)

def symmetry(points, most=12):
    """(M, E[a^M]) for the smallest M whose moment E[a^M] is not zero: the constellation's rotational symmetry (2 BPSK,
    4 QPSK and square QAM, 8 8PSK); (0, 0) when none up to `most` (a constellation with no phase to read)"""
    p = np.asarray(points, dtype=np.complex128)
    scale = np.mean(np.abs(p))
    for m in range(1, most + 1):
        mom = np.mean(p**m)
        if abs(mom) > 1e-6*scale**m:
            return m, complex(mom)
    return 0, 0j
