"""LTE cell search and frame timing (3GPP TS 36.211 6.11): a stream of samples to a stream of whole half-frames

Every 5 ms the last symbol of slot 0 of subframes 0 and 5 carries the primary synchronization signal (PSS: a Zadoff-Chu
root that names N_ID2 = 0..2) on the 62 central subcarriers, and the symbol before it the secondary (SSS: two m-sequences
that name N_ID1 = 0..167 and which half-frame it is). The cell is 3 N_ID1 + N_ID2. FDD, normal cyclic prefix.

Acquiring (on the first `acquire` seconds): the PSS at 1.92 MS/s over a grid of carrier offsets (`offset` +- `span`), its
correlation folded on 5 ms; then each PSS at the full rate, the fractional offset from the cyclic prefix, and the SSS
of each, which votes on N_ID1 and labels the half-frames. Tracking: each half-frame's PSS is found again near where the
last one put it (an alpha-beta loop on its position: the sample clock drifts) and the cyclic prefix trims the offset.

Out come the samples from the first subframe 0 on, carrier offset taken off, cut into half-frames of exactly 5 ms at the
nominal rate (a sample dropped or repeated between them as the clock drifts), so the grid after it sees subframes at
fixed places: subframe n of the output is subframe n mod 10 of a radio frame. When the PSS is lost for `lose`
half-frames it is acquired again, and zeros stand in for the half-frames missed, so that still holds. The cell goes in
the stream's `est["cell"]`.
"""
from types import SimpleNamespace

import numpy as np

from ...base import Stage
from ...base.types import Category, Rep
from ...signal.filters import convolve, correlate, low_pass
from . import phy
from ...sequence import zadoff_chu

ROOTS = (25, 29, 34) # N_ID2 = 0, 1, 2

def pss(u):
    """The PSS of root u: Zadoff-Chu of length 63 without its middle element (it would sit on DC)"""
    return np.delete(zadoff_chu(u, 63), 31)

def bins62(nfft):
    return np.concatenate((np.arange(-31, 0), np.arange(1, 32))) % nfft

def pss_time(u, nfft):
    X = np.zeros(nfft, complex)
    X[bins62(nfft)] = pss(u)
    return np.fft.ifft(X)

def _mseq(taps):
    x = [0, 0, 0, 0, 1]
    for i in range(26):
        x.append(sum(x[i + t] for t in taps) % 2)
    return 1 - 2*np.array(x)

S_T, C_T, Z_T = _mseq((2, 0)), _mseq((3, 0)), _mseq((4, 2, 1, 0))

def sss(n_id1, n_id2, half):
    """The 62 SSS values in subframe 0 (half 0) or 5 (half 1)"""
    qp = n_id1//30
    q = (n_id1 + qp*(qp + 1)//2)//30
    mp = n_id1 + q*(q + 1)//2
    m0 = mp % 31
    m1 = (m0 + mp//31 + 1) % 31
    n = np.arange(31)
    s0, s1 = S_T[(n + m0) % 31], S_T[(n + m1) % 31]
    c0, c1 = C_T[(n + n_id2) % 31], C_T[(n + n_id2 + 3) % 31]
    z0, z1 = Z_T[(n + m0 % 8) % 31], Z_T[(n + m1 % 8) % 31]
    d = np.empty(62)
    if half == 0:
        d[0::2], d[1::2] = s0*c0, s1*c1*z0
    else:
        d[0::2], d[1::2] = s1*c0, s0*c1*z1
    return d

_SSS = {}
def sss_table(n_id2):
    """(336, 62): every (N_ID1, half) hypothesis, N_ID1 major"""
    if n_id2 not in _SSS:
        _SSS[n_id2] = np.array([sss(n1, n_id2, h) for n1 in range(168) for h in (0, 1)])
    return _SSS[n_id2]


class LteCell(Stage):
    """LTE cell search and half-frame timing (see the module)

    offset: where the carrier is expected [Hz from the centre]; span, resolution: the offsets searched around it [Hz]
    acquire: seconds of samples the acquisition looks at; lose: half-frames without a PSS before acquiring again
    gate: the PSS's normalized correlation below which a half-frame has none"""
    CATEGORY = Category.SYNC
    CONTROLS = {}

    def __init__(self, offset=0.0, span=50e3, resolution=5e3, acquire=0.04, lose=20, gate=0.2, name="cell"):
        self.offset, self.span, self.resolution = float(offset), float(span), float(resolution)
        self.acquire, self.lose, self.gate, self.name = float(acquire), int(lose), float(gate), name

    def __str__(self):
        return f"LteCell({self.offset/1e3:+g} kHz)"

    def takes(self, encode=True):
        return {Rep.SAMPLES}

    def new_state(self, meta):
        return SimpleNamespace(buf=None, base=0, lock=None, est={"locked": False, "acquired": 0, "lost": 0, "half_frames": 0})

    # ---- acquisition

    def _search(self, x, fs, nfft):
        """(N_ID2, offset [Hz], PSS position mod 5 ms at the full rate, metric) from the decimated fold. The samples are
        moved by `offset`, filtered to the PSS's +-465 kHz and the span, and taken to 1.92 MS/s once; each offset tried is
        then a turn of the decimated samples"""
        ratio = int(round(fs/1.92e6))
        period = 9600
        y0 = convolve(x*phy.ramp(len(x), 0.0, self.offset/fs), low_pass(129, 0.47e6 + self.span + 50e3, fs).filt, "same")[::ratio]
        td = np.arange(len(y0))/1.92e6
        tmpl = [pss_time(u, 128) for u in ROOTS]
        best = None
        for d in np.arange(-self.span, self.span + self.resolution/2, self.resolution):
            y = y0*np.exp(-2j*np.pi*d*td)
            for r, tp in enumerate(tmpl):
                c = np.abs(correlate(y, tp))**2
                m = len(c)//period*period
                if m == 0:
                    continue
                fold = c[:m].reshape(-1, period).mean(0)
                score = fold.max()/max(np.median(fold), 1e-30)
                if best is None or score > best[3]:
                    best = (r, self.offset + d, int(np.argmax(fold))*ratio, score)
        return best

    def _find_pss(self, y, guess, tmpl, half):
        """(position of the PSS's useful start, normalized correlation) within +-half samples of guess, or None"""
        a = guess - half
        seg = y[max(a, 0):a + 2*half + len(tmpl)]
        if a < 0 or len(seg) < 2*half + len(tmpl):
            return None
        c = np.abs(correlate(seg, tmpl))
        j = int(np.argmax(c))
        e = np.sqrt(np.sum(np.abs(seg[j:j + len(tmpl)])**2)*np.sum(np.abs(tmpl)**2))
        return a + j, float(c[j]/max(e, 1e-30))

    def _cp_offset(self, y, start, nfft, fs, n=70):
        """Residual carrier offset [Hz] from the cyclic prefixes of n symbols from a slot start (+- 7.5 kHz)"""
        cp0, cp = phy.cp_lengths(nfft)
        acc, pos = 0j, start
        for s in range(n):
            c = cp0 if s % 7 == 0 else cp
            if pos < 0 or pos + c + nfft > len(y):
                break
            acc += np.sum(y[pos:pos + c]*np.conj(y[pos + nfft:pos + nfft + c]))
            pos += c + nfft
        return float(-np.angle(acc)/(2*np.pi*nfft/fs)) if acc else 0.0

    def _sss(self, y, p, nfft, n_id2):
        """(N_ID1, half) of the SSS before the PSS whose useful part starts at p (FDD), and its metric"""
        cp0, cp = phy.cp_lengths(nfft)
        t6 = p - cp//4 # (a little early: inside the prefix; the same for both symbols, so the channel takes the ramp)
        t5 = t6 - (cp + nfft)
        if t5 < 0 or t6 + nfft > len(y):
            return None
        b = bins62(nfft)
        Y6, Y5 = np.fft.fft(y[t6:t6 + nfft])[b], np.fft.fft(y[t5:t5 + nfft])[b]
        h = Y6/pss(ROOTS[n_id2])
        z = Y5*np.conj(h)/np.maximum(np.abs(h)**2, 1e-12)
        m = np.abs(sss_table(n_id2) @ z)/np.sqrt(np.sum(np.abs(z)**2)*62)
        i = int(np.argmax(m))
        return i//2, i % 2, float(m[i])

    def _acquire(self, x, fs, nfft):
        """The lock from a buffer of samples, or None"""
        found = self._search(x, fs, nfft)
        if found is None or found[3] < 8.0: # (chance is ~3-5)
            return None
        n_id2, off, p0, _ = found
        period = 5*phy.subframe_len(nfft)
        y = x*phy.ramp(len(x), 0.0, off/fs)
        tmpl = pss_time(ROOTS[n_id2], nfft)
        pos = []
        guess = p0
        while guess + nfft + 64 < len(y):
            r = self._find_pss(y, guess, tmpl, 64 if not pos else 8)
            if r is not None and r[1] >= self.gate:
                pos.append(r[0])
                guess = r[0] + period
            else:
                guess += period
        if len(pos) < 2:
            return None
        pss_off = phy.symbol_starts(nfft)[6]
        off += float(np.median([self._cp_offset(y, p - pss_off, nfft, fs) for p in pos]))
        y = x*phy.ramp(len(x), 0.0, off/fs)
        votes = [self._sss(y, p, nfft, n_id2) for p in pos]
        votes = [(p, v) for p, v in zip(pos, votes) if v is not None]
        if not votes:
            return None
        n1s = [v[0] for _, v in votes]
        n_id1 = max(set(n1s), key=n1s.count)
        if n1s.count(n_id1) < max(2, len(n1s)//2 + 1):
            return None
        # The half each PSS is: the SSS's label, made to alternate (the majority of the agreeing ones decides the phase)
        k0 = [(round((p - pos[0])/period) + v[1]) % 2 for p, v in votes if v[0] == n_id1]
        par = int(sum(k0)*2 > len(k0))
        first = next((p for p in pos if (round((p - pos[0])/period) + par) % 2 == 0 and p >= pss_off), None) # Of a subframe 0
        if first is None: # (whose half-frame starts in the buffer)
            return None
        return SimpleNamespace(n_id1=n_id1, n_id2=n_id2, pci=3*n_id1 + n_id2, f=off, start=float(first - pss_off),
                               period=float(period), half=0, miss=0, tmpl=tmpl, metric=0.0)

    # ---- the stream

    def step(self, x, st, meta):
        fs = meta.Fs
        nfft = int(round(fs/phy.SCS))
        if nfft not in phy.N_RB or abs(nfft*phy.SCS - fs) > 1:
            raise ValueError(f"LteCell: {fs/1e6:g} MS/s is not a multiple of 1.92 MS/s up to 30.72")
        x = np.asarray(x)
        if st.buf is None:
            st.buf, st.base = x[:0].astype(np.complex64), meta.pos
        st.buf = np.concatenate((st.buf, x.astype(np.complex64)))
        L = 5*phy.subframe_len(nfft)
        pss_off = int(phy.symbol_starts(nfft)[6])
        out = []
        while True:
            if st.lock is None:
                n = int(self.acquire*fs)
                if len(st.buf) < max(n, 2*L + 4096):
                    break
                lock = self._acquire(st.buf[:n].astype(np.complex128), fs, nfft)
                if lock is None: # Nothing here: keep the last half-frame's worth (a PSS may straddle) and look again
                    keep = len(st.buf) - L
                    st.buf, st.base = st.buf[keep:], st.base + keep
                    continue
                lock.start += st.base
                if st.est.get("next") is not None: # Acquired again: zeros for the half-frames missed (so many it keeps the half)
                    gap = max(int(round((lock.start - st.est["next"])/L)), 0)
                    gap += (gap + st.est["half"]) % 2
                    out.append(np.zeros(gap*L, np.complex64))
                st.lock = lock
                st.est.update(locked=True, acquired=st.est["acquired"] + 1, pci=lock.pci, n_id1=lock.n_id1, n_id2=lock.n_id2)
            lk = st.lock
            a = int(round(lk.start)) - st.base
            if a < 0: # (behind the buffer: the clock ran fast past a cut; start at what is here)
                lk.start, a = float(st.base), 0
            need = a + L + pss_off + nfft + 16
            if len(st.buf) < need + 16:
                if meta.flush and len(st.buf) >= a + L: # The stream's last half-frame: out as it is (no PSS after it to track)
                    out.append(st.buf[a:a + L]*phy.ramp(L, lk.f*(a + st.base)/fs, lk.f/fs))
                    st.buf, st.base = st.buf[a + L:], st.base + a + L
                    st.est["half_frames"] += 1
                break
            o = min(a, 16) # y[o] is the half-frame's first sample
            y = (st.buf[a - o:need]*phy.ramp(need - a + o, lk.f*(a - o + st.base)/fs, lk.f/fs)).astype(np.complex128)
            r = self._find_pss(y, o + pss_off, lk.tmpl, 6)
            if r is not None and r[1] >= self.gate:
                e = r[0] - (o + pss_off)
                lk.start += 0.3*e
                lk.period += 0.02*e
                lk.miss, lk.metric = 0, r[1]
            else:
                lk.miss += 1
            a2 = int(round(lk.start)) - st.base
            o2 = o + (a2 - a)
            half = y[o2:o2 + L] if 0 <= o2 and o2 + L <= len(y) else y[o:o + L]
            out.append(half.astype(np.complex64))
            lk.f += 0.5*self._cp_offset(y, o2 if 0 <= o2 else o, nfft, fs)
            lk.half ^= 1
            lk.start += lk.period
            st.est.update(half_frames=st.est["half_frames"] + 1, cfo=lk.f, ppm=(lk.period/L - 1)*1e6, metric=lk.metric,
                          next=lk.start, half=lk.half)
            drop = max(int(lk.start) - st.base - 64, 0)
            st.buf, st.base = st.buf[drop:], st.base + drop
            if lk.miss >= self.lose:
                st.lock = None
                st.est.update(locked=False, lost=st.est["lost"] + 1)
        meta.stream.est[self.name] = {k: v for k, v in st.est.items() if k not in ("next", "half")}
        return np.concatenate(out) if out else x[:0].astype(np.complex64)
