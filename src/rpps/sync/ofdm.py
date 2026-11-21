"""OFDM symbol timing and fractional carrier frequency from the cyclic prefix (van de Beek, Sandell & Borjesson 1997)"""
from types import SimpleNamespace

import numpy as np

from ..base import Stage
from . import track


class CPSync(Stage):
    """OFDM symbols found by their cyclic prefix: samples in, the same samples out from the start of a symbol (its prefix),
    whole symbols of n_fft + n_cp, with the fractional carrier frequency offset taken off. wave.OFDMRx follows it

        carriers = samples + rp.sync.CPSync(64, 16) + rp.wave.OFDMRx(64, 16)

    First class (README.dev.md, Sync methods): block estimation over windows at absolute stream positions, `window` seconds
    (at least `min_symbols` OFDM symbols). In a window, the correlation of every sample with the one n_fft later, summed over
    a prefix's length and folded over the window's symbols, peaks where the prefixes start: gamma(t) = sum r[k] r*[k + N]
    over the prefix at t. The ML metric |gamma(t)| - rho * energy(t)/2 (rho = snr/(snr + 1)) takes the timing; the angle of
    gamma there is the carrier frequency, -angle/(2 pi) of a subcarrier spacing (|offset| under half a spacing: an integer
    offset is for the pilots or a preamble). Both are tracked across windows (rpps.sync.track): timing modulo a symbol, so a
    clock some ppm off is followed, the output keeping whole symbols by dropping or repeating one prefix sample when the
    tracked start moves by one; the frequency taken off is a phase line evaluated per sample. A window whose peak is weaker
    than `gate` (|gamma| against the prefix's energy: 1 for a clean signal, about 1/sqrt(symbols) on noise) is skipped and the
    loops coast: the gate is on the metric against what noise gives (|gamma| / energy times sqrt(symbols * prefix): about 1
    on noise, and its peak over the L candidate starts about sqrt(2 ln L)), so it holds for any window length.
    Estimates in stream.est[name]: timing (samples into a symbol), cfo (subcarrier spacings), metric, latency.
    """
    CONTROLS = {}
    ACQUIRE = 4 # Windows the loops run wide while they pull in

    def __init__(self, n_fft=64, n_cp=16, window=1e-3, min_symbols=16, snr=10.0, bw=0.25, gate=4.0, name="ofdm"):
        self.N, self.cp = int(n_fft), int(n_cp)
        self.window, self.min_symbols, self.rho = window, int(min_symbols), snr/(snr + 1.0)
        self.bw, self.gate, self.name = bw, gate, name

    def __str__(self):
        return f"CPSync({self.N}, {self.cp})"

    def new_state(self, meta):
        return SimpleNamespace(buf=np.zeros(0, np.complex64), base=None, k=None, W=None, t=track.new(), f=track.new(),
                               next=None, phase=0.0, metric=0.0)

    def _estimate(self, seg, a0):
        """(timing modulo L at absolute position a0's frame, frequency in subcarrier spacings, metric) of a window"""
        N, cp = self.N, self.cp
        L = N + cp
        x = seg.astype(np.complex128)
        c = x[:-N]*np.conj(x[N:]) # r[k] r*[k + N]
        e = 0.5*(np.abs(x[:-N])**2 + np.abs(x[N:])**2)
        cs, es = np.concatenate(([0], np.cumsum(c))), np.concatenate(([0], np.cumsum(e)))
        g, en = cs[cp:] - cs[:-cp], es[cp:] - es[:-cp] # sums over a prefix's length, from every start
        n = len(g)//L
        if n < 1:
            return None
        G, E = g[:n*L].reshape(n, L).sum(axis=0), en[:n*L].reshape(n, L).sum(axis=0) # folded over the window's symbols
        m = np.abs(G) - self.rho*E
        th = int(np.argmax(m))
        metric = float(np.abs(G[th])/max(E[th], 1e-30))
        z = metric*np.sqrt(n*cp) - np.sqrt(2*np.log(L)) # over what noise's best candidate reaches
        return (a0 + th) % L, float(-np.angle(G[th])/(2*np.pi)), metric, z

    def step(self, x, st, meta):
        N, cp = self.N, self.cp
        L = N + cp
        W = max(int(round(self.window*meta.Fs)), self.min_symbols*L)
        out = []
        for a0, b, seg in track.windows(st, x, meta, W):
            c = (a0 + b)/2
            est = self._estimate(seg, a0) if len(seg) >= 2*L else None
            if est is not None:
                th, eps, st.metric, z = est
                if z >= self.gate:
                    bw = self.bw if st.t.windows >= self.ACQUIRE else max(self.bw, 0.6)
                    track.update(st.t, th, c, L, bw)
                    track.update(st.f, eps, c, 1.0, bw)
                else:
                    track.miss(st.t, c)
                    track.miss(st.f, c)
            if st.t.started:
                out.append(self._symbols(st, b, L))
        keep = st.k*st.W if st.next is None else min(st.k*st.W, st.next)
        track.keep(st, keep)
        meta.stream.est[self.name] = dict(timing=st.t.value % L if st.t.started else None,
                                          cfo=float(track.at(st.f, st.base)) if st.f.started else None,
                                          metric=st.metric, latency=st.W/meta.Fs)
        return np.concatenate(out).astype(np.complex64) if out else np.zeros(0, np.complex64)

    def _symbols(self, st, upto, L):
        """Whole symbols whose start is before `upto` and whose samples are all held, frequency taken off"""
        end = st.base + len(st.buf)
        if st.next is None: # The first symbol start at or after what is held
            th = track.at(st.t, st.base) % L
            st.next = st.base + int(round((th - st.base) % L))
        out = []
        while st.next < upto and st.next + L <= end:
            th = track.at(st.t, st.next) % L # Where the tracked timing puts this symbol's start (modulo a symbol)
            d = (th - st.next) % L
            d = d - L if d > L/2 else d
            if abs(d) >= 1: # The clock moved by a sample: start one later or earlier (inside the prefix: nothing lost)
                st.next += int(np.sign(d))
                if st.next + L > end:
                    break
            f = track.at(st.f, st.next)/self.N # cycles a sample
            ph = st.phase + 2*np.pi*f*np.arange(L) # the phase taken off goes on from symbol to symbol (no jump)
            out.append(st.buf[st.next - st.base:st.next - st.base + L]*np.exp(-1j*ph))
            st.phase = float((st.phase + 2*np.pi*f*L) % (2*np.pi))
            st.next += L
        return np.concatenate(out) if out else np.zeros(0, np.complex64)
