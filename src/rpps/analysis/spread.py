"""Direct-sequence spread spectrum below the noise: the code period, the chip rate, the code, and the data under it

    p = rp.analysis.spread.period(x, Fs)                      # code period (s), from autocorrelation fluctuations
    f = rp.analysis.cyclic.features(x, Fs)                    # the chip rate is the |x|^2 line (f.baud)
    c = rp.analysis.spread.code(x, Fs, chip_rate, p.chips)    # the code's chips (up to sign and a cyclic shift)
    y = rp.analysis.spread.despread(c.chips_rx, c.code)       # the data symbols, the code's gain back

A DSSS signal spreads each data symbol over a code of N chips: under the noise, its spectrum shows nothing. Its
autocorrelation over a window R(tau) = sum x(t) x*(t + tau) has, at lags of a whole code period, a part from the code
repeating, signed by the data: averaged it cancels, but its square does not. So the variance of R_w(tau) over many short
windows w (Burel's fluctuation method, the deterministic part taken off: a pulse's own correlation and its ringing are
the same in every window) peaks at multiples of the period, while noise alone gives it flat.

With the chip rate (rpps.analysis.cyclic: chips are symbols to it) the signal is taken at one sample a chip; cut into
windows of the period, the windows are +-code (a symbol each) plus noise when they start at a symbol boundary: their
covariance's leading eigenvector is the code (Bouder & Burel). Detection goes further under the noise than recovery: the
period's fluctuation peak gains with the record, the chip rate needs the |x|^2 line (searched for in a narrow window once
the period is known, still a few dB short: a 127-chip code over 600k samples is detected at -13 dB, recovered at -11). Not aligned, it splits into two eigenvectors, each the code
over part of the window and zero elsewhere: the leading one's support gives the misalignment, and windows from there give
the code.
"""
import numpy as np

from . import cyclic

class Period:
    """Result of `period`: lag (samples), fine (the lag between samples, from the peak's shape), seconds, chips (when
    chip_rate is given), contrast (the peak over the
    fluctuations' median at other lags: noise alone stays near 1 + a few sigma), significant"""
    __slots__ = ("lag", "fine", "seconds", "chips", "contrast", "significant", "fluct", "z")
    def __init__(self, **kw):
        for k in self.__slots__:
            setattr(self, k, kw.get(k))

    def __str__(self):
        c = f" ({self.chips} chips)" if self.chips else ""
        sig = "" if self.significant else ", not significant"
        return (f"code period {self.seconds*1e6:.2f} us = {self.lag} samples{c}, contrast {self.contrast:.2f} "
                f"(z {self.z:.1f}){sig}")

    __repr__ = __str__

def fluctuations(x, max_lag, window):
    """Variance over windows of R_w(tau), tau = 0 .. max_lag (R_w the autocorrelation of window w): E|R_w|^2 - |E R_w|^2.
    A spread code's part is signed by the data, so its mean is 0 and it is all variance; a signal's own correlation (the
    pulse, its ringing) is the same in every window: all mean, taken off"""
    x = np.asarray(x, dtype=np.complex128)
    nw = (len(x) - max_lag)//window
    p2 = np.zeros(max_lag + 1)
    m = np.zeros(max_lag + 1, dtype=np.complex128)
    N = 1 << int(np.ceil(np.log2(window + max_lag)))
    for i in range(nw):
        seg = x[i*window:i*window + window + max_lag]
        a = np.fft.fft(seg[:window], N)
        b = np.fft.fft(seg, N)
        r = np.fft.ifft(np.conj(a)*b)[:max_lag + 1] # sum_t x*(t) x(t + tau) over the window
        p2 += np.abs(r)**2
        m += r
    nw = max(nw, 1)
    return p2/nw - np.abs(m/nw)**2

def period(x, Fs, max_period=None, window=None, chip_rate=None, min_lag=None, z_min=6.0):
    """The code period of a DSSS signal from autocorrelation fluctuations (see the module). max_period: longest tried
    (samples, default len/64). window: samples a window (default 2 max_period: the code's part adds a symbol at a time,
    so short windows, more of them, are what helps). min_lag: shortest (default 4 Fs / the occupied bandwidth: the
    pulse's own correlation).
    The peak's contrast over the other lags is about 1 + T SNR^2 (T the period in samples, SNR per sample): a long code
    is found further under the noise. Significance: (contrast - 1) sqrt(windows) over z_min"""
    x = np.asarray(x, dtype=np.complex128)
    max_period = int(max_period or len(x)//64)
    window = int(window or 2*max_period)
    if min_lag is None: # the pulse's own correlation lasts a few times Fs / bandwidth (under the noise: unknown, 8)
        _, bw, _ = cyclic.occupied(x, Fs)
        min_lag = int(max(8, 4*Fs/bw)) if bw > Fs/64 else 8
        min_lag = min(min_lag, max_period//4)
    F = fluctuations(x, max_period, window)
    nw = (len(x) - max_period)//window
    base = np.median(F[min_lag:])
    G = F/base
    G[:min_lag] = 0
    z = (G - 1)*np.sqrt(max(nw, 1))
    # A code period is a peak standing clear of the lags around it; a signal's own correlation decays from lag 0
    _, bw_, _ = cyclic.occupied(x, Fs)
    d = int(max(2, 2*np.ceil(Fs/max(bw_, Fs/8))))
    left = np.concatenate((np.full(d, np.inf), F[:-d]))
    right = np.concatenate((F[d:], np.full(d, np.inf)))
    peak = F > 1.5*np.minimum(left, right) # above at least one side by half again: not on a slope from lag 0
    peak &= F >= np.maximum(np.concatenate(([0], F[:-1])), np.concatenate((F[1:], [0])))
    z = np.where(peak, z, 0.0)
    top = z.max()
    lag = int(np.flatnonzero(z >= 0.7*top)[0]) if top > 0 else 0 # the first lag standing out: its multiples do too
    chips = int(round(lag/Fs*chip_rate)) if chip_rate else None
    if 0 < lag < len(F) - 1: # the peak between samples: a parabola through it and its neighbours
        a, b, c = F[lag - 1], F[lag], F[lag + 1]
        den = a - 2*b + c
        frac = float(lag + 0.5*(a - c)/den) if abs(den) > 1e-300 else float(lag)
    else:
        frac = float(lag)
    return Period(lag=lag, fine=frac, seconds=lag/Fs, chips=chips, contrast=float(G[lag]), significant=bool(z[lag] > z_min), z=float(z[lag]),
                  fluct=F)

class Code:
    """Result of `code`: code (+-1 chips, one period), offset (chips: where a symbol starts in the chip stream),
    chips_rx (the chip-rate samples, from `offset`), gain (leading eigenvalue over the next: how clear)"""
    __slots__ = ("code", "offset", "chips_rx", "gain")
    def __init__(self, **kw):
        for k in self.__slots__:
            setattr(self, k, kw.get(k))

    def __str__(self):
        bits = "".join("1" if c > 0 else "0" for c in self.code)
        return f"{len(self.code)}-chip code {bits[:64]}{'...' if len(bits) > 64 else ''}, symbols start at chip " \
               f"{self.offset}, eigen gap {self.gain:.1f}"

    __repr__ = __str__

def _eig(V):
    R = V.conj().T @ V/len(V)
    w, U = np.linalg.eigh(R)
    return w[::-1], U[:, ::-1]

def code(x, Fs, chip_rate, chips, beta=0.35):
    """The spreading code (`chips` long) of a DSSS signal at `chip_rate` (see the module). The carrier must be off to
    well under the symbol rate (chip_rate / chips) or the windows turn: take cyclic.features' cfo off first"""
    s, _ = cyclic.symbol_instants(x, Fs, chip_rate, beta)
    P = int(chips)
    n = len(s)//P
    if n < 32:
        raise ValueError(f"{n} periods: need 32 or more")
    V = s[:n*P].reshape(n, P)
    w, U = _eig(V)
    e = np.abs(U[:, 0])
    # The leading eigenvector is the code over one run of the window (cyclically) and ~0 elsewhere: where it starts
    big = e > 0.5*e.max()
    starts = np.flatnonzero(big & ~np.roll(big, 1))
    if len(starts) and not big.all():
        # the run's start is where symbols start, unless the run is the shorter part: then the other eigenvector's
        lengths = [(np.flatnonzero(~np.roll(big, -st)[1:])[0] + 1 if (~np.roll(big, -st)).any() else P) for st in starts]
        st = int(starts[int(np.argmax(lengths))])
        offset = st if max(lengths) >= P/2 else (st + max(lengths)) % P
    else:
        offset = 0
    s2 = s[offset:]
    n2 = len(s2)//P
    V2 = s2[:n2*P].reshape(n2, P)
    w2, U2 = _eig(V2)
    c = U2[:, 0]
    c = c*np.exp(-1j*np.angle(np.sum(c))) # one phase for all: real chips +-1
    code_ = np.where(c.real >= 0, 1, -1)
    return Code(code=code_, offset=int(offset), chips_rx=s2, gain=float(w2[0]/max(w2[1], 1e-30)))

def despread(chips_rx, code_):
    """Symbols: each period of chip samples correlated with the code (the processing gain back)"""
    P = len(code_)
    n = len(chips_rx)//P
    return (np.asarray(chips_rx)[:n*P].reshape(n, P) @ np.asarray(code_, dtype=float))/P

def chip_rate(x, Fs, lag, min_sps=2.0, pfa=1e-3, fluct=None, fine=None):
    """The chip rate given the code period (lag, samples, maybe a sample off): |x|^2 of a code repeating every period has
    lines at multiples of 1/period; the chip rate is the strongest significant one (the pulse's line) near such a
    multiple, at most Fs / min_sps; without one (far under the noise), from the width of the fluctuation peak (`fluct`,
    Period.fluct, and `fine`, Period.fine, for the narrow search): a chip. Returns (chip rate Hz, chips a period, the period refined: chips / chip rate, s)"""
    x = np.asarray(x, dtype=np.complex128)
    ls = [l for l in cyclic.lines(np.abs(x)**2, Fs, 2*Fs/lag, Fs/min_sps, pfa, top=16) if l.significant]
    best = None
    for l in sorted(ls, key=lambda l: -l.strength):
        k = round(l.freq*lag/Fs)
        if k >= 2 and abs(l.freq - k*Fs/lag) < 1.2*k*Fs/lag**2 + 0.05*Fs/lag: # a sample of lag error moves it k Fs/lag^2
            best = (l.freq, k)
            break
    if best is None:
        if fluct is None:
            raise ValueError("no |x|^2 line at a multiple of the code rate (pass the fluctuations for the peak's width)")
        # Under the noise the line goes first: the fluctuation peak at the period is a chip wide (the code's
        # correlation moved by a chip is gone), so its half-height width is a chip
        F = np.asarray(fluct, dtype=float)
        lo = max(lag - lag//4, 1)
        seg = F[lo:lag + lag//4]
        base = np.median(F[8:])
        h = seg - base
        top = h.max()
        above = np.flatnonzero(h >= top/2)
        width = float(above[-1] - above[0] + 1) if len(above) else 1.0
        k0 = lag/max(width, 1.0)
        # the width is the pulse's (about 0.75 of a chip for root-raised cosine chips): among the harmonics of the
        # period near it, the chip rate is the one |x|^2 has the most power at, line or not
        z = np.abs(x)**2
        z = (z - z.mean())*np.hanning(len(z))
        N = 1 << int(np.ceil(np.log2(4*len(z))))
        S = np.abs(np.fft.rfft(z, N))**2
        ks = np.arange(max(2, int(0.5*k0)), int(2.0*k0) + 2)
        bins = np.round(ks*Fs/lag/(Fs/N)).astype(int)
        ok = bins < len(S) - 3
        ks, bins = ks[ok], bins[ok]
        # a lag error e moves harmonic k by k Fs e / lag^2: e a sample for an integer lag, a quarter for a fine one
        err = 0.25 if fine else 1.2
        lag = fine or lag
        bins = np.round(ks*Fs/lag/(Fs/N)).astype(int)
        win = int(np.ceil(err*ks.max()*Fs/lag**2/(Fs/N))) + 1
        pw = np.array([S[max(b_ - win, 0):b_ + win + 1].max() for b_ in bins])
        # ... and of the strongest few, the one whose code comes out clearest (the leading eigenvalue's lead): at the
        # wrong chip rate the windows hold no fixed code
        # The period is known to a sample, so harmonic k only to k Fs/lag^2: a rate that far off slips the chips by
        # hundreds over the record and no window holds the code. Its line, searched for in that small window only, is
        # found far under the threshold a whole-band search needs: the rate to a fraction of a bin
        f_ = np.fft.rfftfreq(N, 1/Fs)
        best = None
        for k in ks[np.argsort(-pw)[:5]]:
            b0 = int(round(k*Fs/lag/(Fs/N)))
            lo_, hi_ = max(b0 - win, 1), min(b0 + win + 1, len(S) - 1)
            j = lo_ + int(np.argmax(S[lo_:hi_]))
            a, b, c = np.log(S[j - 1:j + 2] + 1e-300)
            d = 0.5*(a - c)/(a - 2*b + c) if abs(a - 2*b + c) > 1e-12 else 0.0
            R = (j + d)*Fs/N
            try:
                g = code(x, Fs, R, int(k)).gain
            except ValueError:
                continue
            if best is None or g > best[0]:
                best = (g, R, int(k))
        if best is None:
            k = int(ks[int(np.argmax(pw))])
            return k*Fs/lag, k, lag/Fs
        _, R, k = best
        return float(R), k, k/R
    R, k = best
    return float(R), int(k), k/R
