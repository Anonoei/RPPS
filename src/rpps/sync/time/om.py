from types import SimpleNamespace

import numpy as np

from ...base import Stage
from .. import track

def cubic(x, t):
    """x (complex) at fractional positions t: cubic Lagrange through x[i-1 .. i+2], i = floor(t)"""
    i = np.floor(t).astype(np.int64)
    u = t - i
    c = (-u*(u - 1)*(u - 2)/6, (u + 1)*(u - 1)*(u - 2)/2, -(u + 1)*u*(u - 2)/2, (u + 1)*u*(u - 1)/6)
    return c[0]*x[i - 1] + c[1]*x[i] + c[2]*x[i + 1] + c[3]*x[i + 2]

class OerderMeyr(Stage):
    """Symbol timing by square-law block estimation (Oerder & Meyr), tracked across windows: samples at `sps` in (after
    their matched filter), one symbol each out

        symbols = matched + rp.sync.time.OerderMeyr()

    First class (README.dev.md, Sync methods). Windows sit at absolute stream positions [k*W, (k+1)*W): W is `window`
    seconds, at least `min_symbols` symbols (the square-law line is weak: at roll-off 0.2 it is about 3% of the power, so it
    wants a few thousand samples). Each window's timing phase is the angle of sum |x[n]|^2 exp(-j 2 pi n / sps), and a
    block-rate loop (rpps.sync.track) carries the phase and the clock's rate (ppm: the transmitter's clock fast by that much) from window to window, so a clock some ppm
    off is followed and each symbol's instant comes from the line, not a step per window. A window whose line is weaker
    than `gate` (its SNR, |X| / sum|x|^2 * sqrt(n): noise alone is about 1) is skipped, the loop coasting through it (a gap
    between bursts). Symbols are interpolated (cubic) from the samples; any sps, fractional too, and any block cut.

    Latency: a window, as `latency` (seconds) in stream.est[name] with the phase, the rate (ppm) and the line's SNR.
    Nothing comes out before the first window that passes the gate.
    """
    CONTROLS = {}
    ACQUIRE = 4 # Windows the loop runs wide (bandwidth 0.6) while it pulls in, then `bw`

    def __init__(self, sps=None, window=1e-3, min_symbols=2048, bw=0.25, gate=2.0, name="timing"):
        self.sps, self.window, self.min_symbols, self.bw, self.gate, self.name = sps, window, min_symbols, bw, gate, name

    def __str__(self):
        return "OerderMeyr"

    def new_state(self, meta):
        return SimpleNamespace(buf=np.zeros(0, np.complex128), base=None, k=None, W=None, m=None, tr=track.new(), snr=0.0)

    def step(self, x, st, meta):
        sps = float(self.sps or meta.sps)
        if sps < 2:
            raise ValueError(f"OerderMeyr needs at least 2 samples per symbol, got {sps}")
        if st.base is None:
            st.base = meta.pos
            st.W = max(int(round(self.window*meta.Fs)), int(np.ceil(self.min_symbols*sps)))
            st.k = meta.pos//st.W
        st.buf = np.concatenate((st.buf, np.asarray(x, dtype=np.complex128)))
        end = st.base + len(st.buf)
        W, out = st.W, []
        while True:
            a, b = st.k*W, (st.k + 1)*W
            if b > end:
                if not (meta.flush and a < end):
                    break
                b = end
            a0 = max(a, st.base)
            seg = st.buf[a0 - st.base:b - st.base]
            if len(seg) >= 16*sps:
                p = np.abs(seg)**2*np.exp(-2j*np.pi*np.arange(a0, b)/sps)
                X = np.sum(p)
                st.snr = float(np.abs(X)/max(np.sum(np.abs(seg)**2), 1e-30)*np.sqrt(len(seg)))
                if st.snr >= self.gate:
                    tau = (-np.angle(X)*sps/(2*np.pi)) % sps
                    rate = None
                    if not st.tr.started: # The clock's rate to start from: the two halves of the window
                        h = len(p)//2
                        d = track.wrap(-np.angle(np.sum(p[h:])/np.sum(p[:h]))*sps/(2*np.pi), sps)
                        rate = d/h
                    bw = self.bw if st.tr.windows >= self.ACQUIRE else max(self.bw, 0.6) # Wide while it pulls in
                    track.update(st.tr, tau, (a0 + b)/2, sps, bw, rate)
                else:
                    track.miss(st.tr, (a0 + b)/2)
            if st.tr.started:
                out.append(self._symbols(st, a0, min(b, end - 3), sps))
            st.k += 1
            if b >= end:
                break
        # Keep the window being gathered, and what the next symbol's interpolation needs
        keep = st.k*W
        if st.m is not None:
            keep = min(keep, int(np.floor(self._time(st, st.m, sps))) - 2)
        keep = min(max(keep, st.base), end)
        st.buf, st.base = st.buf[keep - st.base:], keep
        tr = st.tr
        meta.stream.est[self.name] = dict(phase=tr.value % sps if tr.started else None, ppm=-tr.rate*1e6 if tr.started else None,
                                          snr=st.snr, windows=tr.windows, missed=tr.missed, latency=W/meta.Fs)
        meta.op.Fs = meta.Fs/sps
        meta.op.sps = 1
        return np.concatenate(out).astype(np.complex64) if out else np.zeros(0, np.complex64)

    @staticmethod
    def _time(st, m, sps):
        """Instant of symbol m: t = theta(t) + m*sps, theta the tracked line"""
        tr = st.tr
        return (m*sps + tr.value - tr.rate*tr.at)/(1 - tr.rate)

    def _symbols(self, st, a, b, sps):
        """The symbols whose instants fall before b (the window's end, or what the interpolation can reach)"""
        tr = st.tr
        if st.m is None:
            st.m = int(np.ceil((a + 1 - track.at(tr, a))/sps))
        hi = (b*(1 - tr.rate) - tr.value + tr.rate*tr.at)/sps # m*sps < b(1 - r) - v + r*c
        m = np.arange(st.m, max(int(np.ceil(hi)), st.m))
        if not len(m):
            return np.zeros(0, np.complex128)
        t = self._time(st, m, sps)
        ok = t - 1 >= st.base # (only before the first window, never after)
        m, t = m[ok], t[ok]
        st.m = int(m[-1]) + 1 if len(m) else st.m
        return cubic(st.buf, t - st.base) if len(t) else np.zeros(0, np.complex128)
