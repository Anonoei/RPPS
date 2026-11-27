"""LTE cell search and frame timing (3GPP TS 36.211 6.11): a stream of samples to a stream of whole half-frames

Every 5 ms two symbols carry the synchronization signals on the 62 central subcarriers: the primary (PSS: a Zadoff-Chu
root that names N_ID2 = 0..2) and the secondary (SSS: two m-sequences that name N_ID1 = 0..167 and which half-frame it
is). The cell is 3 N_ID1 + N_ID2. Where the two are says the duplex mode: FDD has the PSS in the last symbol of slot 0
of subframes 0 and 5 and the SSS the symbol before; TDD the PSS in symbol 2 of subframes 1 and 6 and the SSS in the last
symbol of the subframes before. Normal cyclic prefix.

Acquiring (on the first `acquire` seconds): the PSS at 1.92 MS/s over a grid of carrier offsets (`offset` +- `span`), its
correlation folded on 5 ms; then each PSS at the full rate, the fractional offset from the cyclic prefix, and the SSS
of each, which votes on N_ID1 and the duplex mode and labels the half-frames. A cell found has its synchronization
signals subtracted from the samples and the search runs again about its carrier offset, so weaker cells on the same
carrier are found too (`search` lists them; `pci` says which one to receive, else the strongest). Two cells of one
N_ID2 with the same frame timing share their PSS, and the weaker is then seldom told apart. Tracking: each half-frame's PSS is found
again near where the last one put it (an alpha-beta loop on its position: the sample clock drifts) and the cyclic prefix
trims the offset.

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


def _pss_offset(duplex, nfft):
    """Where the PSS's useful part starts in a half-frame that begins with subframe 0 or 5"""
    st = phy.symbol_starts(nfft)
    return int(st[6]) if duplex == "fdd" else phy.subframe_len(nfft) + int(st[2])

def _sss_gap(duplex, nfft):
    """Samples from the SSS's useful part to the PSS's"""
    cp0, cp = phy.cp_lengths(nfft)
    return cp + nfft if duplex == "fdd" else 3*(cp + nfft) + cp0 - cp

def _find_pss(y, guess, tmpl, half):
    """(position of the PSS's useful start, normalized correlation) within +-half samples of guess, or None"""
    a = guess - half
    seg = y[max(a, 0):a + 2*half + len(tmpl)]
    if a < 0 or len(seg) < 2*half + len(tmpl):
        return None
    c = np.abs(correlate(seg, tmpl))
    j = int(np.argmax(c))
    e = np.sqrt(np.sum(np.abs(seg[j:j + len(tmpl)])**2)*np.sum(np.abs(tmpl)**2))
    return a + j, float(c[j]/max(e, 1e-30))

def _cp_offset(y, start, nfft, fs, n=70):
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

CANDIDATES = 16 # PSS peaks tried for a cell
SSS_MIN = 0.45 # the SSS correlation of a cell (noise, and a PSS peak at a wrong offset: about 0.3)
SSS_SURE = 0.7 # one no other candidate needs trying against
SAME_CARRIER = 5e3 # [Hz] cells sharing a carrier differ by less (base stations are within 0.05 ppm; Doppler)
CP_SYMBOLS = {"fdd": 70, "tdd": 14} # the symbols from a half-frame's start whose prefixes give the offset (TDD: downlink for sure)

def _sss(y, p, nfft, n_id2, duplex):
    """(N_ID1, half) of the SSS before the PSS whose useful part starts at p, and its metric"""
    cp = phy.cp_lengths(nfft)[1]
    t6 = p - cp//4 # (a little early: inside the prefix; the same for both symbols, so the channel takes the ramp)
    t5 = t6 - _sss_gap(duplex, nfft)
    if t5 < 0 or t6 + nfft > len(y):
        return None
    b = bins62(nfft)
    Y6, Y5 = np.fft.fft(y[t6:t6 + nfft])[b], np.fft.fft(y[t5:t5 + nfft])[b]
    h = Y6/pss(ROOTS[n_id2])
    # Y5 matched to h times each hypothesis, normalized: 1 where Y5 is h times it, whatever the channel's shape
    m = np.abs(sss_table(n_id2) @ (Y5*np.conj(h)))/np.sqrt(max(np.sum(np.abs(Y5)**2)*np.sum(np.abs(h)**2), 1e-30))
    i = int(np.argmax(m))
    return i//2, i % 2, float(m[i])

def _candidates(x, fs, nfft, offset, span, resolution, roots):
    """PSS peaks of the samples, best first: (N_ID2, offset [Hz], position mod 5 ms at the full rate, peak/median).
    The samples are moved by `offset`, filtered to the PSS's +-465 kHz and the span, and taken to 1.92 MS/s once; each
    offset tried is then a turn of the decimated samples"""
    ratio = int(round(fs/1.92e6))
    period = 9600
    y0 = convolve(x*phy.ramp(len(x), 0.0, offset/fs), low_pass(129, 0.47e6 + span + 50e3, fs).filt, "same")[::ratio]
    td = np.arange(len(y0))/1.92e6
    tmpl = [pss_time(u, 128) for u in ROOTS]
    found = []
    for d in np.arange(-span, span + resolution/2, resolution):
        y = y0*np.exp(-2j*np.pi*d*td)
        for r in roots:
            c = np.abs(correlate(y, tmpl[r]))**2
            m = len(c)//period*period
            if m == 0:
                continue
            fold = c[:m].reshape(-1, period).mean(0)
            floor = max(np.median(fold), 1e-30)
            rest = fold.copy()
            for _ in range(3): # a root's peaks: cells of one N_ID2 differ in timing
                j = int(np.argmax(rest))
                score = rest[j]/floor
                if score < 8.0: # (chance is ~3-5)
                    break
                found.append((r, offset + d, j*ratio, float(score)))
                rest[(j + np.arange(-64, 65)) % period] = 0
    found.sort(key=lambda v: -v[3])
    out = []
    for v in found: # The best offset of each peak. A Zadoff-Chu root also peaks, moved in time, at offsets whole
        # subcarriers away, as high as at the true one: those are candidates of their own, the SSS tells them apart
        if not any(v[0] == w[0] and abs(v[1] - w[1]) <= 10e3
                   and min((v[2] - w[2]) % (period*ratio), (w[2] - v[2]) % (period*ratio)) <= 64*ratio for w in out):
            out.append(v)
    return out

def _lock(x, fs, nfft, candidate, duplex, gate):
    """The lock on the cell at a PSS peak (`_candidates`), or None where the SSS does not agree on one"""
    n_id2, off, p0, _ = candidate
    period = 5*phy.subframe_len(nfft)
    y = x*phy.ramp(len(x), 0.0, off/fs)
    tmpl = pss_time(ROOTS[n_id2], nfft)
    pos, corr = [], []
    guess = p0
    while guess + nfft + 64 < len(y):
        r = _find_pss(y, guess, tmpl, 64 if not pos else 8)
        if r is None and not pos: # (a peak at the very start: its next)
            guess += period
            continue
        if r is not None and r[1] >= gate:
            pos.append(r[0])
            corr.append(r[1])
            guess = r[0] + period
        else:
            guess += period
    if len(pos) < 2:
        return None
    # The duplex mode and N_ID1: the layout whose SSS votes agree
    best = None
    for mode in ("fdd", "tdd") if duplex == "auto" else (duplex,):
        votes = [(p, _sss(y, p, nfft, n_id2, mode)) for p in pos]
        votes = [(p, v) for p, v in votes if v is not None]
        if not votes:
            continue
        n1s = [v[0] for _, v in votes]
        n_id1 = max(set(n1s), key=n1s.count)
        n = n1s.count(n_id1)
        if n >= max(2, len(n1s)//2 + 1) and (best is None or n/len(n1s) > best[0]):
            best = (n/len(n1s), mode, n_id1, votes)
    if best is None:
        return None
    _, mode, n_id1, votes = best
    pss_off = _pss_offset(mode, nfft)
    cps = [_cp_offset(y, p - pss_off, nfft, fs, CP_SYMBOLS[mode]) for p in pos if p >= pss_off]
    if cps:
        off += float(np.median(cps))
    y = x*phy.ramp(len(x), 0.0, off/fs)
    votes = [(p, v) for p, v in ((p, _sss(y, p, nfft, n_id2, mode)) for p in pos) if v is not None and v[0] == n_id1]
    if len(votes) < 2:
        return None
    # The half each PSS is: the SSS's label, made to alternate (the majority of the agreeing ones decides the phase)
    k0 = [(round((p - pos[0])/period) + v[1]) % 2 for p, v in votes]
    par = int(sum(k0)*2 > len(k0))
    first = next((p for p in pos if (round((p - pos[0])/period) + par) % 2 == 0 and p >= pss_off), None) # Of a subframe 0
    if first is None: # (whose half-frame starts in the buffer)
        return None
    return SimpleNamespace(n_id1=n_id1, n_id2=n_id2, pci=3*n_id1 + n_id2, duplex=mode, f=off,
                           start=float(first - pss_off), period=float(period), half=0, miss=0, tmpl=tmpl, metric=0.0,
                           corr=float(np.median(corr)), pss=pos, parity=par, pss_off=pss_off,
                           sss=float(np.mean([v[2] for _, v in votes])))

def _cancel(x, fs, nfft, lk):
    """(x with the cell's PSS and SSS taken out, their power): at each of its occurrences, the channel on the 62
    subcarriers (least squares, smoothed so what other cells put there stays) times the signal sent, subtracted"""
    x = np.array(x)
    cp = phy.cp_lengths(nfft)[1]
    b = bins62(nfft)
    idx = np.arange(-cp, nfft) % nfft # a symbol with its prefix, of one period
    smooth = np.ones(9)
    norm = np.convolve(np.ones(62), smooth, "same")
    gap = _sss_gap(lk.duplex, nfft)
    power = []
    for p in lk.pss:
        half = (round((p - lk.pss[0])/lk.period) + lk.parity) % 2
        for at, seq in ((p, pss(ROOTS[lk.n_id2])), (p - gap, sss(lk.n_id1, lk.n_id2, half))):
            if at - cp < 0 or at + nfft > len(x):
                continue
            turn = phy.ramp(nfft + cp, lk.f*(at - cp)/fs, lk.f/fs)
            W = np.fft.fft(x[at:at + nfft]*turn[cp:])
            h = np.convolve(W[b]*np.conj(seq)/np.abs(seq)**2, smooth, "same")/norm
            S = np.zeros(nfft, complex)
            S[b] = h*seq
            x[at - cp:at + nfft] -= np.fft.ifft(S)[idx]*np.conj(turn)
            power.append(float(np.mean(np.abs(h)**2)))
    return x, (float(np.mean(power)) if power else 0.0)

def _cells(x, fs, nfft, offset, span, resolution, duplex, gate, max_cells):
    """The locks on the cells of the samples, strongest first, each taken out before the next is looked for"""
    x = np.asarray(x, complex)
    seen = set()
    carrier = None # the first cell's offset: the others on the carrier are within SAME_CARRIER of it
    while len(seen) < max_cells:
        # The strongest cell must stand out of the whole band at `gate`; one under it, once it is out, by less
        around, wide = (offset, span) if carrier is None else (carrier, SAME_CARRIER)
        candidates = _candidates(x, fs, nfft, around, wide, resolution if carrier is None else SAME_CARRIER/2, (0, 1, 2))
        best = None
        for c in candidates[:CANDIDATES]:
            if c[3] < 0.2*candidates[0][3]:
                break
            lk = _lock(x, fs, nfft, c, duplex, gate if carrier is None else min(gate, 0.05))
            if (lk is not None and lk.pci not in seen and lk.sss >= SSS_MIN and (best is None or lk.sss > best.sss)
                    and (carrier is None or abs(lk.f - carrier) <= SAME_CARRIER)):
                best = lk
                if lk.sss >= SSS_SURE:
                    break
        if best is None:
            return
        x, best.power = _cancel(x, fs, nfft, best)
        seen.add(best.pci)
        carrier = best.f if carrier is None else carrier
        yield best

def search(x, fs, offset=0.0, span=50e3, resolution=5e3, duplex="auto", acquire=0.04, max_cells=8, gate=0.2):
    """The cells on a carrier, strongest first, from the start of x

    A list of dicts: pci, duplex ("fdd" or "tdd"), offset [Hz from the centre], start (the first sample of a subframe 0),
    sss (the mean SSS correlation, 1: clean), level_db (the synchronization signals' power against the strongest's).
    offset, span, resolution: the carrier offsets searched; acquire: seconds looked at; gate: the strongest cell's PSS
    correlation at least"""
    fs = float(fs)
    nfft = _nfft(fs)
    x = np.asarray(x[:int(acquire*fs)], complex)
    out = []
    for lk in _cells(x, fs, nfft, float(offset), float(span), float(resolution), duplex, gate, int(max_cells)):
        out.append({"pci": lk.pci, "duplex": lk.duplex, "offset": float(lk.f), "start": int(round(lk.start)), "sss": lk.sss,
                    "power": lk.power})
    top = max((c["power"] for c in out), default=1.0) or 1.0
    for c in out:
        c["level_db"] = float(10*np.log10(max(c.pop("power"), 1e-30)/top))
    return out

def _nfft(fs):
    nfft = int(round(fs/phy.SCS))
    if nfft not in phy.N_RB or abs(nfft*phy.SCS - fs) > 1:
        raise ValueError(f"LTE: {fs/1e6:g} MS/s is not 1.92 MS/s times a power of two or 12 up to 30.72")
    return nfft


class LteCell(Stage):
    """LTE cell search and half-frame timing (see the module)

    offset: where the carrier is expected [Hz from the centre]; span, resolution: the offsets searched around it [Hz]
    acquire: seconds of samples the acquisition looks at; lose: half-frames without a PSS before acquiring again
    gate: the PSS's normalized correlation below which a half-frame has none
    pci: the cell to receive (-1: the strongest); duplex: "fdd", "tdd" or "auto" (the SSS says)"""
    CATEGORY = Category.SYNC
    CONTROLS = {}

    def __init__(self, offset=0.0, span=50e3, resolution=5e3, acquire=0.04, lose=20, gate=0.2, pci=-1, duplex="auto",
                 name="cell"):
        if duplex not in ("auto", "fdd", "tdd"):
            raise ValueError(f"LteCell: duplex {duplex!r} (auto, fdd, tdd)")
        self.offset, self.span, self.resolution = float(offset), float(span), float(resolution)
        self.acquire, self.lose, self.gate, self.name = float(acquire), int(lose), float(gate), name
        self.pci, self.duplex = int(pci), duplex

    def __str__(self):
        return f"LteCell({self.offset/1e3:+g} kHz{'' if self.pci < 0 else f', cell {self.pci}'})"

    def takes(self, encode=True):
        return {Rep.SAMPLES}

    def new_state(self, meta):
        return SimpleNamespace(buf=None, base=0, lock=None, est={"locked": False, "acquired": 0, "lost": 0, "half_frames": 0})

    def _acquire(self, x, fs, nfft):
        """The lock from a buffer of samples, or None"""
        n = 1 if self.pci < 0 else 8
        for lk in _cells(x, fs, nfft, self.offset, self.span, self.resolution, self.duplex, self.gate, n):
            if self.pci < 0 or lk.pci == self.pci:
                # A cell under another correlates less over the whole band: tracked at half what it was found at
                lk.gate = self.gate if self.pci < 0 else min(self.gate, 0.5*lk.corr)
                return lk
        return None

    # ---- the stream

    def step(self, x, st, meta):
        fs = meta.Fs
        nfft = _nfft(fs)
        x = np.asarray(x)
        if st.buf is None:
            st.buf, st.base = x[:0].astype(np.complex64), meta.pos
        st.buf = np.concatenate((st.buf, x.astype(np.complex64)))
        L = 5*phy.subframe_len(nfft)
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
                st.est.update(locked=True, acquired=st.est["acquired"] + 1, pci=lock.pci, n_id1=lock.n_id1, n_id2=lock.n_id2,
                              duplex=lock.duplex)
            lk = st.lock
            a = int(round(lk.start)) - st.base
            if a < 0: # (behind the buffer: the clock ran fast past a cut; start at what is here)
                lk.start, a = float(st.base), 0
            pss_off = lk.pss_off
            need = a + L + pss_off + nfft + 16
            if len(st.buf) < need + 16:
                if meta.flush and len(st.buf) >= a + L: # The stream's last half-frame: out as it is (no PSS after it to track)
                    out.append(st.buf[a:a + L]*phy.ramp(L, lk.f*(a + st.base)/fs, lk.f/fs))
                    st.buf, st.base = st.buf[a + L:], st.base + a + L
                    st.est["half_frames"] += 1
                break
            o = min(a, 16) # y[o] is the half-frame's first sample
            y = (st.buf[a - o:need]*phy.ramp(need - a + o, lk.f*(a - o + st.base)/fs, lk.f/fs)).astype(np.complex128)
            r = _find_pss(y, o + pss_off, lk.tmpl, 6)
            if r is not None and r[1] >= lk.gate:
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
            lk.f += 0.5*_cp_offset(y, o2 if 0 <= o2 else o, nfft, fs, CP_SYMBOLS[lk.duplex])
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
