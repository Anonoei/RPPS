"""Equalizers: `symbols + rp.sync.equalize.BlockLS(qpsk)` solves the taps per window (first class);
`symbols + rp.sync.equalize.CMA(11)` and LMS adapt them symbol by symbol (second class)

Each makes one output per symbol from `sps` input samples per symbol (1, or 2 for a fractionally spaced equalizer).
The taps carry across blocks, so any block length gives the same stream.
"""
from types import SimpleNamespace

import numpy as np

from .. import _accel
from ..base import Stage

class _Equalizer(Stage):
    __slots__ = ("taps", "mu", "sps")
    def __init__(self, taps=11, mu=0.01, sps=None):
        """taps: filter length. mu: step size (smaller is steadier, larger tracks faster).
        sps: input samples per symbol (None: what the data says, rounded)"""
        if taps < 1:
            raise ValueError("taps must be at least 1")
        self.taps, self.mu, self.sps = int(taps), mu, sps

    def new_state(self, meta):
        w = np.zeros(self.taps, dtype=np.complex64)
        w[self.taps//2] = 1.0 # Start as a pass-through at the middle tap
        sps = self.sps or max(1, meta.isps)
        return SimpleNamespace(w=w, tail=np.zeros(self.taps - 1, dtype=np.complex64), next=self.taps - 1, sps=sps, n=0)

    def error(self, y, st):
        """Update direction for output y: w += mu * error * conj(x)"""
        raise NotImplementedError()

    def _native_mode(self):
        """0 CMA, 1 LMS: the error functions rpps._native.equalize knows (a subclass with its own runs in Python)"""
        err = type(self).error
        return 0 if err is CMA.error else 1 if err is LMS.error else None

    def step(self, samples, st, meta):
        N, sps, mu = self.taps, st.sps, self.mu
        buf = np.concatenate((st.tail, samples.astype(np.complex64, copy=False)))
        out = []
        n, w = st.next, st.w
        mode = self._native_mode() if _accel.native is not None else None
        if mode is not None:
            pts = getattr(st, "points", None)
            train = getattr(self, "training", None)
            none = np.zeros(0, dtype=np.complex64)
            out, w, n, st.n = _accel.native.equalize(
                buf, np.ascontiguousarray(w, dtype=np.complex64), int(n), int(sps), float(mu), mode, float(getattr(self, "R", 0.0)),
                none if pts is None else np.ascontiguousarray(pts, dtype=np.complex64), pts is not None,
                none if train is None else np.ascontiguousarray(train, dtype=np.complex64), train is not None, int(st.n))
        while mode is None and n < len(buf):
            x = buf[n - N + 1:n + 1][::-1]
            y = np.dot(w, x)
            w = w + mu*self.error(y, st)*np.conj(x)
            out.append(y)
            n += sps
            st.n += 1
        keep = min(n - N + 1, len(buf))
        st.tail, st.next, st.w = buf[keep:], n - keep, w
        meta.op.Fs = meta.Fs/sps
        meta.op.sps = meta.sps/sps
        return np.asarray(out, dtype=np.complex64)

class CMA(_Equalizer):
    """Constant modulus algorithm: blind, for constant-envelope modulations (PSK, FSK) -- it fixes ISI but not rotation

    modulus: the symbol amplitude it steers towards (1.0)
    """
    __slots__ = ("R",)
    def __init__(self, taps=11, mu=0.005, sps=None, modulus=1.0):
        super().__init__(taps, mu, sps)
        self.R = modulus**2

    def error(self, y, st):
        return -y*(abs(y)**2 - self.R)

class LMS(_Equalizer):
    """Least mean squares against a training sequence for the first symbols, then against the nearest point of
    `points` (decision directed). Needs a carrier that is already locked

        eq = rp.sync.equalize.LMS(11, points=rp.symbol.load("QPSK").points, training=known_symbols)

    With neither, the constellation is the detected one (meta.stream.det, see rp.sync.Acquire), so after the
    carrier correction: `... + rp.sync.phase.FFCarrier() + rp.sync.equalize.LMS(7, 0.002)`.
    """
    __slots__ = ("points", "training")
    def __init__(self, taps=11, mu=0.01, sps=None, points=None, training=None):
        super().__init__(taps, mu, sps)
        self.points = None if points is None else np.asarray(points, dtype=np.complex64)
        self.training = None if training is None else np.asarray(training, dtype=np.complex64)

    def new_state(self, meta):
        st = super().new_state(meta)
        st.points = self.points
        return st

    def step(self, samples, st, meta):
        if st.points is None and self.training is None: # The detected constellation, once there is one
            det = meta.stream.det
            if det is None:
                return samples[:0]
            from ..analysis.modulation import candidates  # Late: analysis imports sync
            st.points = candidates([det.mod])[det.mod].astype(np.complex64)
        return super().step(samples, st, meta)

    def error(self, y, st):
        k = st.n
        if self.training is not None and k < len(self.training):
            d = self.training[k]
        elif st.points is not None:
            d = st.points[np.argmin(np.abs(st.points - y))]
        else:
            return 0.0
        return d - y

class BlockLS(Stage):
    """Equalize symbols by least squares over each window, decision directed, with the taps carried from window to window

        symbols = timed + carrier + rp.sync.equalize.BlockLS(rp.symbol.load("QPSK"), taps=7)

    First class (ARCHITECTURE.md, Sync methods): block estimation, where LMS and CMA adapt per symbol. Windows sit at
    absolute stream positions, `window` seconds (at least `min_symbols`). On each, the taps w (symbol spaced, centred:
    output n uses inputs n - taps//2 .. n + taps//2) solve min |X w - d|^2, d the constellation points nearest to X w,
    `iters` times; the window's solution is blended with the last (`blend` of the old: the tracking across windows), and
    that is the filter for the window. It starts as a pass-through and normalizes the power to the constellation's.
    A window that is noise only would pull the taps anywhere: `gate` is the least fraction of decisions within half the
    minimum distance for a window to update the taps. Latency: a window and taps//2 symbols (est: `latency`).
    """
    CONTROLS = {}

    def __init__(self, constellation, taps=7, window=1e-3, min_symbols=1024, iters=3, blend=0.5, gate=0.5, name="equalizer"):
        if taps < 1 or taps % 2 == 0:
            raise ValueError("taps is an odd number")
        self.points = np.asarray(constellation.points, dtype=np.complex128)
        self.dmin = float(constellation.min_distance)
        self.taps, self.window, self.min_symbols = int(taps), window, min_symbols
        self.iters, self.blend, self.gate, self.name = int(iters), float(blend), float(gate), name

    def __str__(self):
        return f"BlockLS({self.taps})"

    def pos_rate(self, decode):
        return 1.0

    def new_state(self, meta):
        h = self.taps//2
        w = np.zeros(self.taps, np.complex128)
        w[h] = 1.0
        return SimpleNamespace(buf=np.zeros(h, np.complex128), base=meta.pos - h, k=None, W=None, w=w, scaled=False,
                               good=0.0, windows=0)

    def _decide(self, y):
        return self.points[np.argmin(np.abs(y[:, None] - self.points[None, :]), axis=1)]

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
            w = st.w
            for _ in range(self.iters):
                y = X @ w
                d = self._decide(y)
                good = float(np.mean(np.abs(y - d) < self.dmin/2))
                if good < self.gate:
                    break
                w = np.linalg.lstsq(X, d, rcond=None)[0]
            else:
                st.w = self.blend*st.w + (1 - self.blend)*w
                st.windows += 1
            st.good = good
            out.append((X @ st.w)[:max(0, min(b, real) - a0)]) # (past the stream's end on a flush: padding)
            st.k += 1
            if meta.flush and b + h >= end:
                break
        keep = min(max(st.k*W - h, st.base), end)
        st.buf, st.base = st.buf[keep - st.base:], keep
        meta.stream.est[self.name] = dict(taps=st.w.copy(), good=st.good, windows=st.windows, latency=(W + h)/meta.Fs)
        return np.concatenate(out).astype(np.complex64) if out else np.zeros(0, np.complex64)
