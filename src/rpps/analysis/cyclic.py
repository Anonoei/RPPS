"""Cyclostationary features: the symbol rate and carrier lines a signal has, and how sure each one is

    f = rp.analysis.cyclic.features(x, Fs)    # x at baseband, a few times wider than the signal
    print(f)                                  # e.g. "baud 1.000 MHz (line 31 dB over the floor, P_fa 1e-12); carrier
                                              #  lines x^4 at +12.3 kHz: QPSK / QAM / APSK"
    f.baud, f.order, f.cfo                    # None where no line is significant: "no rate" is an answer

A linearly modulated signal is cyclostationary: its statistics repeat every symbol. Put through a nonlinearity, it gives
spectral lines a stationary signal (noise, OFDM without its prefix, a carrier with a continuous phase) does not:
  |x|^2    a line at the symbol rate (stronger the larger the roll-off: the line lives in the band edges)
  x^M      a line at M times the carrier offset when the constellation repeats every 360/M degrees (x^2 for BPSK / ASK,
           x^4 for QPSK and QAM, x^8 for 8PSK, x^12 for 4+12 / 4+12+16 APSK), with side lines at that +- the symbol
           rate. Between symbols the pulses overlap and spoil high powers, so once the rate is known the powers are also
           taken of matched-filtered samples at the symbol instants (`symbol_instants`), far more sensitive at small
           roll-off and for x^8 / x^12; noise raised to the 12th power needs Es/N0 near 15 dB or more
  x        a line at the carrier itself if any of it is left (a pilot tone, AM, an unbalanced constellation)
  |d|^2    d = the instantaneous frequency (x[n] conj(x[n-1])): a line at the symbol rate for a constant-envelope signal
           (FSK, MSK, CPM) whose |x|^2 is flat

Each line is tested against the spectrum around it. Without a line, the power of a periodogram bin is exponential about its
local mean, so a bin `r` times over the floor happens with probability exp(-r); over K bins searched, the chance noise
alone gives one that high is P_fa = 1 - (1 - exp(-r))^K; the floor is itself estimated from about m bins, which makes the
tail (1 + r/m)^-m instead. A line is significant when P_fa is under `pfa`. The floor is the
local median over ln 2 (robust to the lines themselves).

The band the signal occupies is estimated first (Welch PSD over its floor) and everything else is filtered off (a Wiener
mask), so the noise outside the signal does not dilute the lines.

`scf` is the spectral correlation function (FFT accumulation method) for a picture of all cyclic frequencies at once.
"""
import numpy as np

from ..math.interp import parabolic_offset

HARMONICS = 12 # The highest harmonic of a rate a line is taken to be (lines weaken; any R has a high one near any f)
CONSTANT_ENVELOPE = 1.2 # E|x|^4/(E|x|^2)^2 under this: FM, FSK, MSK, CPM (filtered PSK is 1.3 and more)
KINDS = ("|x|^2", "x", "x^2", "x^4", "x^8", "|d|^2")
ORDER_NAMES = {1: "a residual carrier / pilot", 2: "BPSK / ASK (2-fold)", 4: "QPSK / QAM / APSK (4-fold)",
               8: "8PSK (8-fold)", 12: "12-fold (an APSK outer ring)", 16: "16PSK (16-fold)"}

class Line:
    """A spectral line of a nonlinearity of the signal. kind: one of KINDS. freq: where it is (Hz; for x^M, M times the
    carrier offset). strength: its power over the local floor (linear). pfa: chance noise gives one that strong in the
    band searched. significant: pfa under the threshold asked for"""
    __slots__ = ("kind", "freq", "strength", "pfa", "significant")
    def __init__(self, kind, freq, strength, pfa, significant):
        self.kind, self.freq, self.strength, self.pfa, self.significant = kind, freq, strength, pfa, significant

    @property
    def db(self):
        return 10*np.log10(max(self.strength, 1e-30))

    def __str__(self):
        return f"{self.kind} line at {self.freq:.1f} Hz, {self.db:.1f} dB over the floor (P_fa {self.pfa:.1e})"

    __repr__ = __str__

class Features:
    """Result of `features`.

    baud: the symbol rate (Hz) from the strongest significant |x|^2 line, else from |d|^2 (constant envelope), else None.
    baud_line: that Line. baud_from: "|x|^2" or "|d|^2". order: the smallest M whose x^M spectrum has a significant line
    (1, 2, 4, 8), None without one. cfo: carrier offset (Hz) from that line (ambiguous by Fs/M). carrier: {M: Line} of
    every significant x^M line. confirmed: the x^M line has side lines at +- baud (the rate and carrier belong to one
    signal). envelope: E|x|^4 / (E|x|^2)^2 (1 constant envelope, 1.3-1.8 filtered PSK/QAM, 2 Gaussian). bandwidth: of the
    occupied band (Hz). lines: every line tested."""
    __slots__ = ("baud", "baud_line", "baud_from", "order", "cfo", "carrier", "confirmed", "envelope", "bandwidth",
                 "lines", "pfa")
    def __init__(self, **kw):
        for k in self.__slots__:
            setattr(self, k, kw.get(k))

    @property
    def linear(self):
        """A symbol rate seen in |x|^2: a linearly modulated carrier (PSK, QAM, APSK, ASK)"""
        return self.baud is not None and self.baud_from == "|x|^2"

    def __str__(self):
        out = []
        if self.baud is None:
            out.append(f"no symbol-rate line (P_fa {self.pfa:g}): not a linear modulation at a rate in the band searched,"
                       " or too few samples for its roll-off")
        else:
            l = self.baud_line
            out.append(f"baud {self.baud:.1f} Hz from {self.baud_from} ({l.db:.1f} dB over the floor, P_fa {l.pfa:.1e})")
        if self.order is None:
            out.append("no carrier line at any power tried (more-fold symmetry, CPM, OFDM, noise, or too little SNR for it)")
        else:
            c = "" if not self.confirmed else ", side lines at +- baud"
            out.append(f"carrier line x^{self.order} at {self.cfo:+.1f} Hz: {ORDER_NAMES.get(self.order, f'{self.order}-fold')}{c}")
        out.append(f"envelope {self.envelope:.2f}, bandwidth {self.bandwidth:.1f} Hz")
        return "; ".join(out)

    __repr__ = __str__

def _floor(p, width):
    """Local mean of exponential bins: running median over `width` bins / ln 2"""
    width = max(5, int(width) | 1)
    if len(p) <= width:
        return np.full_like(p, np.median(p)/np.log(2))
    h = width//2
    pad = np.concatenate((p[h:0:-1], p, p[-2:-h-2:-1]))
    from numpy.lib.stride_tricks import sliding_window_view
    # Medians over a strided set of windows, interpolated: a full running median is slow for long spectra
    step = max(1, width//8)
    centers = np.arange(0, len(p), step)
    med = np.median(sliding_window_view(pad, width)[centers], axis=1)
    return np.interp(np.arange(len(p)), centers, med)/np.log(2)

def lines(z, Fs, lo, hi, pfa=1e-3, top=4, pad=2, width=None, exclude=None):
    """Significant spectral lines of the sequence `z` (sampled at Fs) between lo and hi Hz (negative frequencies allowed for
    a complex z): up to `top` Lines, strongest first, each at least `exclude` Hz from a stronger one (default 4 bins)"""
    z = np.asarray(z)
    n = len(z)
    if n < 64:
        return []
    w = np.hanning(n)
    z = (z - np.mean(z))*w if np.isrealobj(z) else z*w
    N = pad*n
    P = np.abs(np.fft.fft(z, N))**2/np.sum(w**2)
    f = np.fft.fftfreq(N, 1/Fs)
    order = np.argsort(f)
    P, f = P[order], f[order]
    width = width or 128*pad + 1
    fl = _floor(P, width)
    m = max(4.0, 0.43*width/pad) # Effective exponential samples behind a median floor (Hann: ~1.5 bins each)
    r = P/np.maximum(fl, 1e-300)
    band = (f >= lo) & (f <= hi)
    if not band.any():
        return []
    K = max(1, int(band.sum()//pad)) # independent bins searched
    idx = np.flatnonzero(band)
    exclude = 4*Fs/n if exclude is None else exclude
    out = []
    rb = r[idx].copy()
    for _ in range(top):
        j = int(np.argmax(rb))
        if rb[j] <= 0:
            break
        k = idx[j]
        d = parabolic_offset(*np.log(P[k-1:k+2] + 1e-300)) if 0 < k < N - 1 else 0.0
        fk = f[k] + d*Fs/N
        s = float(r[k])
        q = np.exp(-m*np.log1p(s/m)) # One bin over s: (1 + s/m)^-m, the floor itself being an estimate
        p = float(-np.expm1(K*np.log1p(-min(q, 1 - 1e-16)))) # Any of K bins: 1 - (1 - q)^K
        out.append(Line(None, float(fk), s, p, p < pfa))
        rb[np.abs(f[idx] - f[k]) < exclude] = 0
    return out

def noise_floor(S, q=10, stop_db=50.0):
    """The noise floor of a power spectrum: its q-th percentile, the bins more than `stop_db` under the median left out
    (a channelizer's stopband, zeros: not noise, and far under it)"""
    S = np.asarray(S, dtype=float)
    keep = S > np.median(S)*10**(-stop_db/10)
    return float(np.percentile(S[keep], q)) if keep.any() else float(np.percentile(S, q))

def occupied(x, Fs, nfft=1024):
    """(Wiener mask over FFT bins of x's length, occupied bandwidth Hz, (lowest, highest) frequency over the floor Hz): the
    signal's band estimated from a Welch PSD over its floor (the quietest tenth of the spectrum)"""
    n = len(x)
    nfft = int(min(nfft, 2**int(np.log2(max(n//8, 16)))))
    seg = n//nfft
    S = np.mean(np.abs(np.fft.fft(x[:seg*nfft].reshape(seg, nfft)*np.hanning(nfft), axis=1))**2, axis=0)
    S = np.convolve(np.concatenate((S[-2:], S, S[:2])), np.ones(5)/5, mode="valid") # a little smoothing, circular
    N0 = noise_floor(S)
    sig = np.maximum(S - N0, 0)
    bw = Fs*np.count_nonzero(sig >= N0)/nfft # Bins over 0 dB SNR
    # Wiener gain, never under -20 dB and smoothed over a few bins: a mask with hard zeros or steps makes edges a running
    # floor cannot follow (false lines there)
    band, edges = _edges(sig >= N0, sig, Fs)
    G = np.maximum(sig/np.maximum(S, 1e-300)*band, 0.01) # only the signal's own band: a spur elsewhere would make lines
    G = np.convolve(np.concatenate((G[-5:], G, G[:5])), np.hanning(11)/np.sum(np.hanning(11)), mode="valid")
    G = np.sqrt(G) # Amplitude gain
    fb = np.fft.fftfreq(nfft)
    fx = np.fft.fftfreq(n)
    o = np.argsort(fb)
    mask = np.interp(fx, fb[o], G[o], period=1.0)
    return mask, bw, edges

def _edges(on, power, Fs):
    """The run of `on` bins (FFT order, circular) holding the most power among those wider than a tone: the signal's band,
    not a spur's bin elsewhere. Returns (1 on the run's bins, (lowest, highest) frequency Hz)"""
    n = len(on)
    if not on.any() or on.all():
        return np.ones(n), (-Fs/2, Fs/2)
    wide = max(8, n//50) # a tone (and its window and smoothing) is a few bins; a channel is wider than 2% of itself
    # Close gaps narrower than that: a weak signal's band is ragged at the floor
    on = on.copy()
    off = np.flatnonzero(~on)
    if len(off):
        gaps = np.split(off, np.flatnonzero(np.diff(off) > 1) + 1)
        if len(gaps) > 1 and gaps[0][0] == 0 and gaps[-1][-1] == n - 1: # the gap across the wrap is one gap
            gaps = [np.concatenate((gaps[-1], gaps[0]))] + gaps[1:-1]
        for g in gaps:
            if len(g) < wide:
                on[g] = True
    if on.all():
        return np.ones(n), (-Fs/2, Fs/2)
    start = int(np.flatnonzero(~on)[0]) # a gap: runs never wrap past it
    best, run, total, wide_power = None, [], 0.0, 0.0
    for i in range(start, start + n + 1):
        k = i % n
        if i < start + n and on[k]:
            run.append(i)
            total += power[k]
        elif run:
            key = (len(run) >= wide, total) # a tone is a bin or three: a signal's band is wider
            wide_power += total if len(run) >= wide else 0.0
            if best is None or key > best[0]:
                best = (key, run[0], run[-1])
            run, total = [], 0.0
    (_, held), a, b = best
    if held < 0.5*wide_power: # no one band holds most of it: several signals, or one at the floor
        return np.ones(n), (-Fs/2, Fs/2)
    f = lambda i: ((i % n) - n if (i % n) >= n/2 else (i % n))*Fs/n
    lo = f(a)
    band = np.zeros(n)
    band[np.arange(a, b + 1) % n] = 1.0
    return band, (lo, lo + (b - a)*Fs/n)

def _tag(ls, kind):
    for l in ls:
        l.kind = kind
    return ls

def comb_rate(lines, Fs, lo=0.0, explain=0.85):
    """The symbol rate behind a set of significant lines (cyclic.Line) of a nonlinearity of the signal: every line is a
    harmonic k R, or one folded about the sample rate, Fs - k R. The rate is the largest R whose harmonics explain
    explaining the most of the lines' strength (dB over the floor, summed), within (1 - `explain`)/3 of the best: a
    submultiple of R explains them as well but is smaller, a multiple leaves R's own line unexplained. Returns (R, the strongest line on R's comb)"""
    pool = []
    for l in sorted(lines, key=lambda l: -l.strength): # one per frequency
        if all(abs(l.freq - m.freq) > 1e-3*l.freq for m in pool):
            pool.append(l)
    total = sum(l.db for l in pool)
    def on(f, R):
        tol = 2e-3*f + 1e-6*Fs
        for g in (f, Fs - f):
            k = round(g/R)
            if 1 <= k <= HARMONICS and abs(g - k*R) < tol: # the 61st harmonic of some R lands near anything
                return True
        return False
    cands = []
    for l in pool:
        for d in range(1, 7):
            R = l.freq/d
            if R < lo:
                break
            cands.append((sum(m.db for m in pool if on(m.freq, R))/total, R))
    top = max(c[0] for c in cands)
    # The most explained; of those about as good (a submultiple explains as much), the largest
    R = max(R for share, R in cands if share >= top - (1 - explain)/3)
    near = [l for l in pool if abs(l.freq - R) < 2e-3*R + 1e-6*Fs]
    if near: # The rate from its own line where there is one (a better estimate than a harmonic's divided)
        R = near[0].freq
    line = max((l for l in pool if on(l.freq, R)), key=lambda l: l.strength)
    return R, line

def symbol_instants(x, Fs, baud, beta=0.35, taps=8):
    """Matched-filtered samples at the symbol instants: an RRC at the baud (roll-off `beta`, a guess: a mismatch only
    leaves a little ISI), timing from the phase of the |y|^2 line (its cyclic component peaks at the instants).
    Returns (symbols, first instant in samples)"""
    from ..signal.filters.shaping import rrc_pulse
    sps = Fs/baud
    span = 8
    t = np.arange(-int(np.ceil(span*sps)), int(np.ceil(span*sps)) + 1)/sps
    h = rrc_pulse(t, beta)
    h = h/np.sqrt(np.sum(h**2))
    nfft = 1 << int(np.ceil(np.log2(len(x) + len(h))))
    y = np.fft.ifft(np.fft.fft(x, nfft)*np.fft.fft(h, nfft))[len(h)//2:len(h)//2 + len(x)]
    n = np.arange(len(y))
    c = np.sum(np.abs(y)**2*np.exp(-2j*np.pi*baud*n/Fs))
    tau = (-np.angle(c)/(2*np.pi)*sps) % sps # c ~ exp(-j 2 pi tau/sps): the line peaks at tau
    k = np.arange(int(np.ceil((REACH - tau)/sps)), int((len(y) - REACH - 1 - tau)//sps))
    from ..math import interp
    return interp.fractional(y, tau + k*sps, taps), tau

REACH = 8

def _at_symbols(x, Fs, baud, orders, pfa):
    """x^M lines of the symbols at their instants, each resolved to one frequency: a line at f in the symbol-rate spectrum
    is at f + j baud in the samples' (the candidate whose x^M correlates best)"""
    s, tau = symbol_instants(x, Fs, baud)
    if len(s) < 64:
        return {}
    s = s/np.sqrt(np.mean(np.abs(s)**2))
    xs = x/np.sqrt(np.mean(np.abs(x)**2))
    n = np.arange(len(x))
    out = {}
    for M in orders:
        ls = lines(s**M, baud, -baud/2, baud/2, pfa, top=1)
        if not ls:
            continue
        l = ls[0]
        l.kind = f"x^{M} at symbols" if M > 1 else "x at symbols"
        j = np.arange(-int(Fs/(2*baud)) - 1, int(Fs/(2*baud)) + 2)
        cand = l.freq + j*baud
        cand = cand[np.abs(cand) <= Fs/2]
        p = xs**M
        l.freq = float(cand[np.argmax([abs(np.sum(p*np.exp(-2j*np.pi*f*n/Fs))) for f in cand])])
        out[M] = l
    return out

def features(x, Fs, lo=None, hi=None, pfa=1e-3, orders=(1, 2, 4, 8, 12), mask=True, max_samples=2**22):
    """Symbol rate and carrier lines of baseband samples x (see the module). lo/hi: the symbol rates searched, Hz
    (default Fs/64 .. Fs/1.5). pfa: false-alarm probability of a line. mask: filter to the occupied band first"""
    x = np.asarray(x, dtype=np.complex128)[:max_samples]
    n = len(x)
    lo = Fs/64 if lo is None else lo
    hi = Fs/1.5 if hi is None else hi
    p2 = np.mean(np.abs(x)**2)
    env = float(np.mean(np.abs(x)**4)/max(p2**2, 1e-300)) if n else 0.0
    bw, edges = Fs, (-Fs/2, Fs/2)
    if mask and n >= 1024:
        G, bw, edges = occupied(x, Fs)
        if bw > 0:
            x = np.fft.ifft(np.fft.fft(x)*G)
    out = dict(lines=[], carrier={}, envelope=env, bandwidth=bw, pfa=pfa)
    if n < 256:
        return Features(**out)
    sq = _tag(lines(np.abs(x)**2, Fs, lo, hi, pfa), "|x|^2")
    d = x[1:]*np.conj(x[:-1])
    d = np.angle(d)
    dd = _tag(lines(d**2, Fs, lo, hi, pfa), "|d|^2")
    out["lines"] += sq + dd
    sig_sq = [l for l in sq if l.significant]
    base = None
    if sig_sq: # The rate behind the lines (the strongest can be a harmonic: rectangular pulses, a burst structure)
        R, _ = comb_rate(sig_sq, Fs, lo)
        base = min(sig_sq, key=lambda l: abs(l.freq - R)) if any(abs(l.freq - R) < 2e-3*R for l in sig_sq) else None
        if base is None:
            base = Line("|x|^2", R, max(l.strength for l in sig_sq), min(l.pfa for l in sig_sq), True)
    sig_dd = [l for l in dd if l.significant]
    disc = None
    if sig_dd:
        R, _ = comb_rate(sig_dd, Fs, lo)
        disc = min(sig_dd, key=lambda l: abs(l.freq - R))
        if abs(disc.freq - R) >= 2e-3*R:
            disc = Line("|d|^2", R, max(l.strength for l in sig_dd), min(l.pfa for l in sig_dd), True)
    if disc is not None and env < CONSTANT_ENVELOPE:
        # FSK / MSK / CPM: |x|^2 is flat on air, and the band filter's ripple puts lines (at twice the rate) in it
        out.update(baud=disc.freq, baud_line=disc, baud_from="|d|^2")
    elif base is not None:
        out.update(baud=base.freq, baud_line=base, baud_from="|x|^2")
    xs = x/np.sqrt(max(p2, 1e-300))
    found = {}
    for M in orders:
        ls = _tag(lines(xs**M, Fs, -Fs/2, Fs/2, pfa, top=3), "x" if M == 1 else f"x^{M}")
        out["lines"] += ls
        if ls and ls[0].significant:
            found[M] = ls[0]
            if out.get("baud"):
                side = [l for l in ls[1:] if l.significant and abs(abs(l.freq - ls[0].freq) - out["baud"]) < 4*Fs/n]
                out["confirmed"] = out.get("confirmed") or (bool(side) and M == min(found))
    if out.get("baud_from") == "|x|^2" and len(found) < len(orders):
        # Between symbols the pulses overlap and spoil high powers; at the symbol instants x^M is the constellation's own
        for M, l in _at_symbols(x, Fs, out["baud"], orders, pfa).items():
            if M not in found:
                out["lines"].append(l)
                if l.significant:
                    found[M] = l
    # A line of x^M at f is a carrier at (f + k Fs)/M for some k: the one in the band, nearest its centre. A line whose
    # carrier would be outside the band is a spur or another signal's, not this one's
    centre, half = (edges[0] + edges[1])/2, (edges[1] - edges[0])/2 + 0.1*bw
    cfo = {}
    for M, l in list(found.items()):
        cands = [(l.freq + k*Fs)/M for k in range(-M, M + 1)]
        cands = [c for c in cands if -Fs/2 <= c <= Fs/2]
        c = min(cands, key=lambda c: abs(c - centre))
        if abs(c - centre) <= half:
            cfo[M] = c
        else:
            del found[M]
    out["carrier"] = dict(sorted(found.items()))
    if found:
        M = min(found)
        out.update(order=M, cfo=cfo[M])
    return Features(**out)

def scf(x, Fs, nfft=64, frames=None, hop=None):
    """Spectral correlation function by the FFT accumulation method (Roberts, Brown & Loomis 1991).

    Returns (f, alpha, S): S[i, j] = |S_x^alpha(f)| at f[i], alpha[j] (Hz), on the grid of channel pairs. nfft: channels
    (frequency resolution Fs/nfft). frames: how many windows of hop samples are used (alpha resolution Fs/(frames hop))"""
    x = np.asarray(x, dtype=np.complex128)
    hop = nfft//4 if hop is None else hop
    P = (len(x) - nfft)//hop + 1 if frames is None else frames
    P = int(2**int(np.log2(max(P, 2))))
    if (P - 1)*hop + nfft > len(x):
        raise ValueError("Too few samples for the frames asked")
    idx = np.arange(nfft)[None, :] + hop*np.arange(P)[:, None]
    X = np.fft.fftshift(np.fft.fft(x[idx]*np.hamming(nfft), axis=1), axes=1) # P x nfft channels
    k = np.arange(-nfft//2, nfft//2)
    X = X*np.exp(-2j*np.pi*k[None, :]*hop*np.arange(P)[:, None]/nfft) # Each channel to baseband (its time reference)
    w = np.hamming(P)[:, None]
    fk = k*Fs/nfft
    Q = P # Output bins per channel pair
    ff, aa, ss = [], [], []
    for a in range(nfft):
        prod = X[:, a:a + 1]*np.conj(X)*w # pairs (a, b) for every b
        Z = np.fft.fftshift(np.fft.fft(prod, axis=0), axes=0)/P # Q x nfft
        q = np.arange(-Q//2, Q//2)*Fs/(hop*P)
        alpha = fk[a] - fk[None, :] + q[:, None]
        f = (fk[a] + fk[None, :])/2 + 0*q[:, None]
        keep = np.abs(q[:, None]) <= Fs/(2*nfft) # The part of each pair's output the pair is responsible for
        keep = np.broadcast_to(keep, Z.shape)
        ff.append(f[keep]); aa.append(alpha[keep]); ss.append(np.abs(Z[keep]))
    return np.concatenate(ff), np.concatenate(aa), np.concatenate(ss)

def alpha_profile(x, Fs, nfft=64, frames=None, bins=None):
    """Max over frequency of |SCF| against cyclic frequency alpha (alpha = 0 removed): (alpha Hz, profile). The symbol rate
    and its multiples stand out for a linear modulation"""
    f, a, s = scf(x, Fs, nfft, frames)
    bins = bins or 4*nfft*4
    edges = np.linspace(-Fs, Fs, bins + 1)
    j = np.clip(np.digitize(a, edges) - 1, 0, bins - 1)
    prof = np.zeros(bins)
    np.maximum.at(prof, j, s)
    c = (edges[:-1] + edges[1:])/2
    zero = np.abs(c) <= Fs/(nfft)
    prof[zero] = 0
    return c, prof
