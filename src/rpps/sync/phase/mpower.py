from types import SimpleNamespace

import numpy as np

from ...base import Stage
from .. import track

class MPower(Stage):
    """Carrier frequency and phase of M-fold symmetric symbols by the M-th power, block estimation tracked across windows

        symbols = timed + rp.sync.phase.MPower(order=4, offset=np.pi)     # QPSK at 45 degrees: its points^4 are -1

    First class (ARCHITECTURE.md, Sync methods). Windows sit at absolute stream positions, `window` seconds (at least
    `min_symbols`). The first window that shows the carrier acquires it: the FFT of y^M (zero padded) gives the
    frequency, |f| < 1/(2M) cycles per symbol, and the phase at its centre. After that each window measures the
    residual phase of y turned back by the tracked line, (angle(sum u^M) - offset) / M with u = y e^{-j phase} / |y|, and a block-rate
    loop (rpps.sync.track) carries phase and frequency on, so the correction is a line per symbol, with no staircase and no
    slips while the drift stays inside the loop. A window whose M-th power line is weaker than `gate` (its SNR: noise
    alone is about 1) is skipped; after `relock` of those in a row the carrier is acquired again from the FFT. Acquiring
    wants `gate` more than the best of the FFT's N frequencies reaches on noise, sqrt(2 ln N): at an Es/N0 where the M-th
    power is lost in noise (DVB-S2 under 0 dB with M = 8) nothing is acquired and the symbols pass, for a preamble and
    pilots to take the phase.

    The phase is known modulo 2 pi / M: a preamble, a differential code or a polarity search takes the rest. order: M (a
    profile may change it with the modulation, live). offset: the angle of the constellation's mean point^M (0 for BPSK,
    8PSK, and DVB-S2's pi/2-BPSK under M = 8; pi for QPSK at 45 degrees under M = 4).
    Until it acquires, symbols pass unchanged. Estimates in stream.est[name]: frequency (cycles per symbol), phase, SNR,
    latency (seconds).
    """
    CONTROLS = {"order": "live"}

    def __init__(self, order=4, offset=0.0, window=1e-3, min_symbols=1024, bw=0.25, gate=3.0, relock=4, name="carrier"):
        self.order, self.offset, self.window, self.min_symbols = int(order), float(offset), window, min_symbols
        self.bw, self.gate, self.relock, self.name = bw, gate, relock, name

    def __str__(self):
        return f"MPower({self.order})"

    def control(self, order=None):
        if order is not None:
            self.order = int(order)

    def pos_rate(self, decode):
        return 1.0

    def new_state(self, meta):
        return SimpleNamespace(buf=np.zeros(0, np.complex128), base=None, k=None, W=None, tr=track.new(), snr=0.0)

    def _acquire(self, seg, n, c):
        M = self.order
        z = _unit(seg)**M
        N = 4*(1 << int(np.ceil(np.log2(max(len(z), 2)))))
        F = np.abs(np.fft.fft(z, N))
        i = int(np.argmax(F))
        a, b_, d = F[(i - 1) % N], F[i], F[(i + 1) % N]
        den = a - 2*b_ + d
        k = i + (0.0 if den == 0 else 0.5*(a - d)/den)
        f = ((k/N + 0.5) % 1.0 - 0.5)/M # cycles per symbol
        s = np.sum(z*np.exp(-2j*np.pi*M*f*(n - c)))
        return f, s, z

    def step(self, x, st, meta):
        if st.base is None:
            st.base = meta.pos
            st.W = max(int(round(self.window*meta.Fs)), int(self.min_symbols))
            st.k = meta.pos//st.W
        st.buf = np.concatenate((st.buf, np.asarray(x, dtype=np.complex128)))
        end = st.base + len(st.buf)
        W, M, out = st.W, self.order, []
        while True:
            a, b = st.k*W, (st.k + 1)*W
            if b > end:
                if not (meta.flush and a < end):
                    break
                b = end
            a0 = max(a, st.base)
            seg = st.buf[a0 - st.base:b - st.base]
            n = np.arange(a0, b)
            c = (a0 + b)/2
            tr = st.tr
            if len(seg) >= 8:
                if not tr.started or tr.missed >= self.relock:
                    f, s, z = self._acquire(seg, n, c)
                    st.snr = float(np.abs(s)/max(np.sum(np.abs(z)), 1e-30)*np.sqrt(len(seg)))
                    # The best of N frequencies: noise alone reaches about sqrt(2 ln N) there, so acquiring wants more
                    N = 4*(1 << int(np.ceil(np.log2(max(len(z), 2)))))
                    if st.snr >= self.gate + np.sqrt(2*np.log(N)):
                        tr.started = False
                        track.update(tr, (np.angle(s) - self.offset)/M, c, rate=2*np.pi*f)
                else:
                    z = _unit(seg*np.exp(-1j*track.at(tr, n)))**M
                    s = np.sum(z)
                    st.snr = float(np.abs(s)/max(np.sum(np.abs(z)), 1e-30)*np.sqrt(len(seg)))
                    if st.snr >= self.gate:
                        e = track.wrap((np.angle(s) - self.offset)/M, 2*np.pi/M)
                        bw = self.bw if tr.windows >= 4 else max(self.bw, 0.6) # Wide while it pulls in
                        track.update(tr, track.at(tr, c) + e, c, 2*np.pi/M, bw)
                    else:
                        track.miss(tr, c)
            out.append(seg*np.exp(-1j*track.at(tr, n)) if tr.started else seg)
            st.k += 1
            if b >= end:
                break
        keep = min(max(st.k*W, st.base), end)
        st.buf, st.base = st.buf[keep - st.base:], keep
        tr = st.tr
        meta.stream.est[self.name] = dict(frequency=tr.rate/(2*np.pi) if tr.started else None,
                                          phase=float(tr.value % (2*np.pi/M)) if tr.started else None, snr=st.snr,
                                          windows=tr.windows, missed=tr.missed, latency=W/meta.Fs)
        return np.concatenate(out).astype(np.complex64) if out else np.zeros(0, np.complex64)

def _unit(y):
    """y with its magnitudes taken off: the phase alone goes into the M-th power (Viterbi & Viterbi with no amplitude
    weight), so a few large noise samples can't make a line, and noise alone gives |sum z| / sqrt(n) about 1"""
    a = np.abs(y)
    return np.where(a > 0, y/np.maximum(a, 1e-30), 0)

def offset_of(constellation, order):
    """The angle of a constellation's mean point^M: what MPower takes off as `offset`"""
    pts = np.asarray(constellation.points)
    return float(np.angle(np.mean(pts**order)))
