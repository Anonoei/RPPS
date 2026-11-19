"""OFDM pilot patterns, blind: which cells of the time-frequency grid carry known values, and the lattice they form

    o = rp.analysis.waveform.ofdm(x, Fs)                 # (or waveform.classify(x, Fs).ofdm): Tu, CP, comb
    p = rp.analysis.ofdm_pilots.discover(x, Fs, o)       # or tu=, tcp= when known
    print(p)    # e.g. "OFDM 1024 x 1/4 CP, 601 used subcarriers: pattern every 4 symbols; scattered pilots every 12th
                #  subcarrier, moving 3 a symbol; 45 continual pilots; pilots 2.5 dB over data"
    p.mask      # (period, K) bool: pilot cells by symbol in the period and subcarrier; p.empty: cells never used
    g = rp.analysis.ofdm_pilots.grid(x, Fs, tu, tcp)     # the grid itself: g.Y (symbols, K), g.on, g.k

The grid: x is resampled so that Tu is a power-of-2 FFT; each symbol's start is the peak of its cyclic-prefix
correlation, searched a little either side of where the last one said (so a longer first CP in a slot, LTE's, is
followed); the fractional carrier offset is the correlation's phase; the FFT window starts a quarter of the CP early.
Subcarriers from the first to the last within floor_db of the signal's plateau are kept; symbols far under the
strongest are off (TDD, bursts).

Pilots are cells whose value is known, so a pilot cell is the same (or follows a sequence that repeats) from one
period of the pattern to the next, and data cells are not: Y[t, k] Y*[t + L, k] summed over t keeps its phase at a pilot
cell and averages away at a data cell. The lag L that repeats the most power (each subcarrier's autocorrelation along
time, by FFT, the magnitudes summed over subcarriers) is the pattern's period (the smallest lag within `share` of the
strongest: DVB-T's scattered pilots 4 symbols, a sequence that changes every symbol and repeats each frame its frame,
LTE's cell reference signal 140). At that lag every cell's coherence |sum Y Y*| / sum |Y|^2 over its symbols in the
period is near 1 for a pilot and about 1/sqrt(n) for data: over max(`coherence`, 5/sqrt(n)) is a pilot. Then:
  scattered  rows (symbols in the period) whose pilots are equally spaced: the spacing, and the offset's step a symbol
  continual  subcarriers that are pilots in nearly every symbol
  empty      cells with almost no power in every period (guard bands, DC, the gaps of a comb)
  boost      pilot over data power
Rows (symbols in the period) are decided as a whole: a lattice of subcarriers (spacing D, offset o) whose coherence
stands over the row's (a t-statistic less what the best of D offsets reaches by chance, holding across the band, not in
one block: a broadcast channel repeating its content is no lattice), strong alone or weaker with the spacing most rows
have; or the whole symbol known (its median coherence high: a preamble, a sync or sounding symbol); single cells only
where enough periods make one cell's coherence telling (irregular continual pilots).
Limits: a long period needs many periods: an 84 ms record of LTE TDD (7 frames of 140 symbols) gives its known symbols
and most of its reference-signal rows (spacing 6, offsets alternating by 3), not all. The pattern is read relative to
the first kept subcarrier (an integer carrier offset shifts it); a common
phase that wanders over the lag (phase noise, a symbol clock off by ppm over a long lag) weakens the coherence (the
per-symbol common phase is taken out against the previous period first).
"""
import numpy as np

from .cpm import interpolate

class Grid:
    """`grid`: Y (symbols, K) complex cells (kept subcarriers only), k (K,) their FFT bins (0 = the carrier, signed),
    starts (symbols,) each symbol's start (samples at fs), on (symbols,) bool, N (FFT size), ncp (CP samples at fs),
    fs (the resampled rate), cfo (Hz: the fractional offset taken out), rho (symbols,) each CP correlation"""
    __slots__ = ("Y", "k", "starts", "on", "N", "ncp", "fs", "cfo", "rho")
    def __init__(self, **kw):
        for k in self.__slots__:
            setattr(self, k, kw.get(k))

def _cp_metric(x, N, ncp):
    """|sum over the CP of x[n+i] x*[n+i+N]| over the mean power of both windows, and the sum itself, for each n"""
    p = x[:-N]*np.conj(x[N:])
    c = np.concatenate(([0], np.cumsum(p)))
    s = c[ncp:] - c[:-ncp]
    e1 = np.concatenate(([0], np.cumsum(np.abs(x)**2)))
    w = e1[ncp:] - e1[:-ncp]
    en = 0.5*(w[:len(s)] + w[N:N + len(s)])
    return np.abs(s)/np.maximum(en, 1e-30), s

def _smooth_starts(starts, good, max_q=16):
    """Symbol starts without the peak picker's jitter (a few samples, which turn every subcarrier by a different
    angle): a straight line through the clear peaks (the symbol clock), plus an offset by symbol in a period Q (a CP
    that is longer at the start of each slot: LTE's); Q (1 .. max_q) is the shortest that leaves no more than 1.2x the
    least residual spread"""
    t = np.arange(len(starts))
    if good.sum() < 4:
        return starts
    tg, sg = t[good], starts[good].astype(np.float64)
    # robust line: medians of slopes between halves, then the median intercept
    h = len(tg)//2
    slope = np.median((sg[h:h + h] - sg[:h])/np.maximum(tg[h:h + h] - tg[:h], 1))
    icpt = np.median(sg - slope*tg)
    res = sg - (icpt + slope*tg)
    best = None
    for q in range(1, max_q + 1):
        off = np.array([np.median(res[tg % q == r]) if np.any(tg % q == r) else 0.0 for r in range(q)])
        spread = float(np.mean(np.abs(res - off[tg % q])))
        best = best or (spread, q, off)
        if spread < best[0]:
            best = (spread, q, off)
    floor = best[0]
    for q in range(1, max_q + 1): # the shortest period about as good
        off = np.array([np.median(res[tg % q == r]) if np.any(tg % q == r) else 0.0 for r in range(q)])
        if float(np.mean(np.abs(res - off[tg % q]))) <= 1.2*floor + 0.25:
            break
    return np.round(icpt + slope*t + off[t % q]).astype(np.int64)

def grid(x, Fs, tu, tcp, floor_db=10.0, off_db=10.0, max_symbols=None):
    """The OFDM time-frequency grid of baseband x at Fs (see the module): Grid. tu, tcp: useful symbol and CP (s)"""
    x = np.asarray(x, dtype=np.complex128)
    N = 1 << int(np.ceil(np.log2(tu*Fs - 1e-6)))
    fs = N/tu
    if abs(fs - Fs) > 1e-9*Fs:
        x = interpolate(x, np.arange(int(len(x)*fs/Fs) - 8)*Fs/fs)
    ncp = max(1, int(round(tcp*fs)))
    Ns = N + ncp
    rho, s = _cp_metric(x, N, ncp)
    if len(rho) < 2*Ns:
        raise ValueError("fewer than two symbols")
    # the first symbol: the strongest peak in the first two periods
    n0 = int(np.argmax(rho[:2*Ns]))
    n0 = n0 % Ns if n0 >= Ns and rho[n0 % Ns] > 0.5*rho[n0] else n0
    d = max(2, int(0.02*N))
    starts, rhos = [], []
    n = n0
    while n + Ns + d < len(rho) and (max_symbols is None or len(starts) < max_symbols):
        lo, hi = max(0, n - d), min(len(rho), n + d + 1)
        k = lo + int(np.argmax(rho[lo:hi]))
        # follow the peak only when it is clear; else keep time
        if rho[k] > 0.3:
            n = k
        starts.append(n)
        rhos.append(float(rho[n]))
        n += Ns
    starts = np.array(starts)
    rhos = np.array(rhos)
    good = rhos > 0.3
    starts = _smooth_starts(starts, good)
    ang = np.angle(np.sum(s[starts[good]])) if good.any() else 0.0
    f = -ang/(2*np.pi*N) # cycles a sample
    cfo = f*fs
    win = starts + ncp - ncp//4
    win = win[win + N <= len(x)]
    idx = win[:, None] + np.arange(N)[None, :]
    seg = x[idx]*np.exp(-2j*np.pi*f*idx)
    Y = np.fft.fftshift(np.fft.fft(seg, axis=1), axes=1)/np.sqrt(N)
    bins = np.arange(N) - N//2
    pw_sym = np.mean(np.abs(Y)**2, axis=1)
    on = pw_sym > np.max(pw_sym)*10**(-off_db/10)
    pk = np.mean(np.abs(Y[on])**2, axis=0)
    live = pk > 1e-6*pk.max() # resampling leaves bins outside the capture's band (nearly) empty
    top = np.percentile(pk[live], 95)
    used = pk > top*10**(-floor_db/10) # within floor_db of the signal's plateau
    if not used.any():
        raise ValueError("no subcarriers over the floor")
    a, b = np.flatnonzero(used)[[0, -1]]
    keep = np.arange(a, b + 1) # contiguous: nulls inside the band (DC, comb gaps) stay, they are part of the pattern
    return Grid(Y=Y[:, keep], k=bins[keep], starts=win, on=on, N=N, ncp=ncp, fs=fs, cfo=cfo, rho=rhos[:len(win)])

class Pilots:
    """`discover`: period (symbols), mask (period, K) pilot cells, empty (period, K) unused cells, coherence (period,
    K), scattered [(row, offset, spacing, count)], step (offset change a symbol of the scattered lattice, None), spacing
    (its subcarrier spacing), continual (kept-subcarrier indices), boost_db, lags [(lag, score)] strongest first,
    grid (Grid), lattices {row: (spacing, offset, t)}: each row's pilot lattice (spacing 1: the whole symbol known)"""
    __slots__ = ("period", "mask", "empty", "coherence", "scattered", "step", "spacing", "continual", "boost_db",
                 "lags", "grid", "lattices")
    def __init__(self, **kw):
        for k in self.__slots__:
            setattr(self, k, kw.get(k))

    def __str__(self):
        g = self.grid
        cp = g.ncp/g.N
        head = f"OFDM {g.N} x {cp:.4g} CP, {self.mask.shape[1]} subcarriers kept"
        if self.period is None:
            return head + ": no repeating cells (no pilots found)"
        bits = [f"pattern every {self.period} symbols"]
        full = [r for r, (D, _, _) in sorted(self.lattices.items()) if D == 1]
        if full:
            bits.append(f"whole symbols known at {_rows(full)}")
        if self.spacing:
            mv = f", moving {self.step} a symbol" if self.step else ""
            rows = [x[0] for x in self.scattered]
            where = f"symbols {_rows(rows)}" if len(rows) < self.period else "every symbol"
            bits.append(f"pilots every {self.spacing} subcarriers{mv} in {where}")
        if len(self.continual):
            bits.append(f"{len(self.continual)} continual pilots")
        if self.boost_db is not None:
            bits.append(f"pilots {self.boost_db:+.1f} dB over data")
        ne = int(self.empty.sum())
        if ne:
            bits.append(f"{ne} empty cells a period")
        return head + ": " + "; ".join(bits)

    __repr__ = __str__

def _rows(rows, most=12):
    r = ", ".join(str(x) for x in rows[:most])
    return r + (f" ... ({len(rows)})" if len(rows) > most else "")

def _lag_scores(Y, on, max_lag):
    """score[L] (L = 0 .. max_lag): sum over subcarriers of |sum_t Y[t] Y*[t+L]| (symbols both on), over the same with
    magnitudes (1 when every cell repeats exactly)"""
    T, K = Y.shape
    Z = np.where(on[:, None], Y, 0)
    m = 1 << int(np.ceil(np.log2(2*T)))
    F = np.fft.fft(Z, m, axis=0)
    R = np.fft.ifft(np.conj(F)*F, axis=0)[:max_lag + 1] # R[L, k] = sum_t Z[t+L] Z*[t]
    A = np.fft.fft(np.abs(Z), m, axis=0)
    Ra = np.real(np.fft.ifft(np.conj(A)*A, axis=0))[:max_lag + 1]
    return np.abs(R).sum(axis=1)/np.maximum(Ra.sum(axis=1), 1e-30)

def _common_phase(Y, P, on):
    """Each symbol's common phase against the symbol P before (the mean of Y[t] Y*[t-P] over subcarriers): taken out
    cumulatively within each residue class so the pattern's cells line up across periods"""
    T = len(Y)
    out = Y.copy()
    for t in range(P, T):
        if on[t] and on[t - P]:
            c = np.sum(out[t]*np.conj(out[t - P]))
            if abs(c) > 0:
                out[t] = out[t]*np.exp(-1j*np.angle(c))
    return out

def discover(x, Fs, o=None, tu=None, tcp=None, max_lag=None, share=0.6, row_coherence=0.6, min_t=7.0, weak_t=3.5, max_spacing=24,
             max_symbols=4000):
    """The pilot pattern of an OFDM signal x (see the module). o: rpps.analysis.waveform.ofdm's result (tu and tcp are
    taken from it; a pilot comb's symbol is comb x its lag, with the CP of the system it names), else tu and tcp (s).
    max_lag: longest pattern period searched (symbols; default a quarter of the symbols, at most 320)"""
    if tu is None:
        if o is None:
            from . import waveform
            o = waveform.ofdm(x, Fs)
            if o is None:
                raise ValueError("not OFDM (rpps.analysis.waveform.ofdm found no cyclic prefix)")
        tu, tcp = _symbol_of(o)
    g = grid(x, Fs, tu, tcp, max_symbols=max_symbols)
    Y = g.Y/np.sqrt(np.mean(np.abs(g.Y[g.on])**2))
    T, K = Y.shape
    max_lag = max_lag or max(2, min(320, T//4))
    sc = _lag_scores(Y, g.on, max_lag)
    # data alone: each subcarrier's sum of T products of random phase is ~sqrt(T) of the T magnitudes'
    base = 1/np.sqrt(max(int(g.on.sum()), 1))
    order = np.argsort(-sc[1:]) + 1
    lags = [(int(L), float(sc[L])) for L in order[:8]]
    top = sc[order[0]]
    if top < 3*base:
        return Pilots(period=None, mask=np.zeros((1, K), bool), empty=np.zeros((1, K), bool),
                      coherence=np.zeros((1, K)), scattered=[], step=None, spacing=None, continual=np.array([], int),
                      boost_db=None, lags=lags, grid=g, lattices={})
    P = int(min(L for L in range(1, max_lag + 1) if sc[L] >= share*top and sc[L] > 3*base))
    Yc = _common_phase(Y, P, g.on)
    num = np.zeros((P, K), np.complex128)
    den = np.zeros((P, K))
    pw = np.zeros((P, K))
    cnt = np.zeros(P)
    for t in range(T - P):
        r = t % P
        if g.on[t] and g.on[t + P]:
            num[r] += Yc[t + P]*np.conj(Yc[t])
            den[r] += 0.5*(np.abs(Yc[t])**2 + np.abs(Yc[t + P])**2)
            pw[r] += np.abs(Yc[t])**2
            cnt[r] += 1
    rows = cnt > 0
    coh = np.where(den > 0, np.abs(num)/np.maximum(den, 1e-30), 0)
    pw = pw/np.maximum(cnt, 1)[:, None]
    level = np.median(pw[rows]) if rows.any() else 0.0
    empty = (pw < 0.05*level) & rows[:, None]
    # single cells: only where enough periods make a cell's coherence alone telling (irregular continual pilots)
    thr = np.maximum(0.9, 5/np.sqrt(np.maximum(cnt, 1)))[:, None]
    mask = (coh > thr) & ~empty & rows[:, None]
    lattices, fits = {}, {}
    for r in np.flatnonzero(rows):
        live = ~empty[r]
        c = coh[r]
        if live.sum() < 8:
            continue
        cl = c[live]
        if np.median(cl) > row_coherence: # a whole symbol known (a preamble, a sync or sounding symbol)
            mask[r] |= live
            lattices[r] = (1, 0, 99.0)
            continue
        lat = _lattice(c, live, max_spacing)
        if lat is not None and lat[2] >= weak_t:
            fits[r] = lat
    # a row's lattice counts when it is strong alone, or weaker but with the spacing most rows found (or a divisor of
    # it, or a multiple: two lattices of one pattern in a symbol read as one at half the spacing, LTE's)
    if fits:
        Ds = [D for D, _, _ in fits.values()]
        common = max(set(Ds), key=Ds.count)
        for r, (D, o, t) in fits.items():
            if t >= min_t or common % D == 0 or D % common == 0:
                k = np.arange(K)
                on_lat = (k % D == o) & ~empty[r]
                hit = np.flatnonzero(on_lat & (coh[r] > 0.5*np.median(coh[r][on_lat])))
                if len(hit):
                    on_lat &= (k >= hit[0]) & (k <= hit[-1]) # between the first and last that hold
                mask[r] |= on_lat
                lattices[r] = (D, o, t)
    nrow = max(int(rows.sum()), 1)
    continual = np.flatnonzero(mask[rows].sum(axis=0) >= 0.9*nrow)
    scat = [(int(r), int(o), int(D), int(np.sum(mask[r]))) for r, (D, o, t) in sorted(lattices.items()) if D >= 2]
    spacing = step = None
    if scat:
        sps = [x[2] for x in scat]
        spacing = max(set(sps), key=sps.count)
        rs = [x for x in scat if x[2] == spacing]
        # the offset's change from one pilot symbol to the next, per symbol between them
        steps = [((b[1] - a[1]) % spacing, b[0] - a[0]) for a, b in zip(rs, rs[1:])]
        one = [d for d, gap in steps if gap == 1]
        if one:
            step = max(set(one), key=one.count)
    data = ~mask & ~empty & rows[:, None]
    boost = None
    if mask.any() and data.any():
        boost = float(10*np.log10(pw[mask].mean()/pw[data].mean()))
    return Pilots(period=P, mask=mask, empty=empty, coherence=coh, scattered=scat, step=step, spacing=spacing,
                  continual=continual, boost_db=boost, lags=lags, grid=g, lattices=lattices)

def _lattice(c, live, max_spacing, quarters=3):
    """The subcarrier lattice (spacing D >= 2, offset o) whose cells' coherence c stands most over the row's (live cells
    only): (D, o, t), t the difference of means over its standard error, less sqrt(2 ln D) (the best of D offsets
    stands out by about that by chance). A lattice must hold across the band (its cells over the row's mean in at least
    `quarters` of the band's four quarters): structure in one block of subcarriers (a broadcast channel repeating its
    content) is not a pilot lattice. None when nothing qualifies"""
    k = np.arange(len(c))
    cl = c[live]
    mu, sd = cl.mean(), cl.std() + 1e-12
    lk = k[live]
    edges = np.linspace(lk[0], lk[-1] + 1, 5)
    best = None
    for D in range(2, max_spacing + 1):
        for o in range(D):
            sel = live & (k % D == o)
            n = int(sel.sum())
            if n < 8:
                continue
            t = (c[sel].mean() - mu)/(sd/np.sqrt(n)) - np.sqrt(2*np.log(D))
            if best is not None and t <= best[2]:
                continue
            held = 0
            for q in range(4):
                sq = sel & (k >= edges[q]) & (k < edges[q + 1])
                if sq.sum() >= 2 and c[sq].mean() > mu + sd/np.sqrt(sq.sum()):
                    held += 1
            if held >= quarters:
                best = (D, o, float(t))
    return best

def _symbol_of(o):
    """(tu, tcp) of an rpps.analysis.waveform.OFDM result: a comb's symbol is comb x its lag, with the shortest CP of the
    table's systems at that Tu"""
    if not o.comb:
        return o.tu, o.tcp
    from .waveform import SYSTEMS
    tu = o.comb*o.tu
    cps = [cp*1e-6 for _, stu, cp in SYSTEMS if abs(stu*1e-6 - tu)/tu < 0.01]
    if not cps:
        raise ValueError(f"a comb of {o.comb} at {o.tu*1e6:.2f} us, but no system with Tu {tu*1e6:.2f} us: give tcp")
    return tu, min(cps)
