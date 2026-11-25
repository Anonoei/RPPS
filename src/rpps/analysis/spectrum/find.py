"""Signals in a capture: centre, width, SNR and when they are on, with a false-alarm rate you set

    sv = rp.analysis.spectrum.find.find(x, Fs, cf=cf)       # a Survey of Signal (the same types as find_signals)
    for s in sv: s.lo, s.hi, s.snr, s.t0, s.t1, s.bursts

How it works, each step a known detector:

1. One spectrogram: Hann frames, |FFT|^2 (P[t, f]; every later step reads only this).
2. The noise floor as a function of frequency (the receiver's passband: tilt, ripple, roll-off), so the spectrum can be
   whitened: Z = P / floor. The band is cut into blocks, each the mean of its noise bins (FCME: bins clearly in signals
   are cut out and the floor fitted again), a block with too little noise left interpolated from its neighbours. The
   floor shows between carriers side by side in their notches: a block whose notches (its least 4-bin means) sit well
   under its fit is held to them. The first guess is a line through the blocks' low percentiles, fitted again without
   the blocks over it (a plateau even half the band wide stays out); the receiver's roll-off at the edges is left out.
3. Detection at a false-alarm rate: under noise the mean of k bins over m frames of Z is Gamma distributed with known
   degrees of freedom (the window's correlation between bins and between overlapping frames counted), so a threshold for
   `pfa` comes from its quantile (Wilson-Hilferty). Widths k = 1, 2, 4... bins are each tested, on means over k bins at
   a stride of k/2 (a GLRT over an unknown width: a weak wide signal is found at a coarse scale, a narrow one at a fine
   one), on the mean over all frames and on the most of each group of frames (bursts, which the average dilutes; a
   max-hold of single frames loses weak steady signals). The test count is split over scales and groups (Bonferroni) and
   the floor's own error, measured on the noise bins, widens each threshold. A high threshold seeds a detection and a
   low one grows it (hysteresis, LAD's double threshold); what a width finds is excised before the next (CLEAN).
4. When it is on: each span's band power frame by frame, gated by the same two thresholds (bursts timed on frames 32
   bins of the span wide); a bursty span is measured on the frames it is on.
5. The signals, read by level (rpps.analysis.spectrum.levels): in each region the detector marked, a line is raised
   from the floor a step of a few noise deviations at a time. What comes up and holds its sides from one step to the
   next is a signal; one that comes up out of a wider one stands on it (`over`; the wider `holds` it: a pedestal, or a
   carrier under a narrower signal); a hump's sides never hold. Two carriers side by side come up as one run and part
   at the notch: the run is only them. Lines (a few bins wide: tones, spurs) are taken out first: no signal.
6. Each signal's edges: where it falls `drop_db` under its top, or halfway to where it came up (the floor, a notch, a
   pedestal) when that is less; it passes the detector at its own width and has an SNR of 3 dB or more (what can't be
   processed is no signal). A Finder keeps each region's reading while it holds (measured again each step).

Effort (rp.set_effort): fast skips frame overlap and tests bursts only in groups of 8 frames; balanced and max overlap
frames by half and test groups of 2 and 8. Every level tests widths up to a quarter of the band: capped
lower, a wide weak signal or a hump is found as fragments, each passing on its own.
"""
import functools

import numpy as np

from ... import _effort
from ...math.mask import runs
from . import levels
from .signal import Burst, Signal, Survey

# --- statistics (no scipy)

@functools.lru_cache(maxsize=1024)
def _norm_isf(p):
    """z with P(N(0,1) > z) = p (Acklam's rational approximation, ~1e-9)"""
    p = float(np.clip(p, 1e-300, 1 - 1e-16))
    a = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02, 1.383577518672690e+02,
         -3.066479806614716e+01, 2.506628277459239e+00)
    b = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02, 6.680131188771972e+01,
         -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00, -2.549732539343734e+00,
         4.374664141464968e+00, 2.938163982698783e+00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00)
    q = min(p, 1 - p)
    if q < 0.02425:
        r = np.sqrt(-2*np.log(q))
        z = (((((c[0]*r + c[1])*r + c[2])*r + c[3])*r + c[4])*r + c[5])/((((d[0]*r + d[1])*r + d[2])*r + d[3])*r + 1)
    else:
        r = q - 0.5
        s = r*r
        z = (((((a[0]*s + a[1])*s + a[2])*s + a[3])*s + a[4])*s + a[5])*r/(((((b[0]*s + b[1])*s + b[2])*s + b[3])*s + b[4])*s + 1)
    return -z if p <= 0.5 else z # z above is the lower quantile of q

def _gamma_isf(p, n):
    """x with P(X > x) = p for X the mean of n unit exponentials (Gamma(n, 1/n)): Wilson-Hilferty"""
    v = 1/(9*float(n))
    return max(1 - v + _norm_isf(p)*np.sqrt(v), 1e-3)**3

@functools.lru_cache(maxsize=64)
def _window_terms(nfft, hop):
    """The Hann window's correlation between bins (lags 1..7) and between frames hop apart (lags 1, 2, ...)"""
    win = np.hanning(nfft)
    w2 = win**2
    e2 = w2.sum()
    rb = np.abs(np.fft.fft(w2)[1:8])**2/e2**2
    lt = np.arange(1, int(np.ceil(nfft/hop)))
    rt = np.array([(win[j*hop:]*win[:nfft - j*hop]).sum()**2/e2**2 for j in lt])
    return rb, rt

@functools.lru_cache(maxsize=4096)
def _dof_cached(nfft, hop, M, k):
    rb, rt = _window_terms(nfft, hop)
    lags = np.arange(1, min(k, 8))
    cb = 1 + 2*np.sum((1 - lags/k)*rb[:len(lags)]) if len(lags) else 1.0
    lt = np.arange(1, min(M, len(rt) + 1))
    ct = 1 + 2*np.sum((1 - lt/M)*rt[:len(lt)]) if len(lt) else 1.0
    return k*M/(cb*ct)

def _dof(nfft, hop, M, k):
    """Degrees of freedom of the mean of k adjacent bins over M frames of a Hann periodogram (unit-mean exponential bins,
    correlated: neighbours through the window's spectrum, overlapping frames through its overlap)"""
    return _dof_cached(int(nfft), int(hop), int(max(M, 1)), int(max(k, 1)))

def _box(z, k):
    """Centred mean of k bins (same length; the ends average what they have)"""
    k = int(k)
    if k <= 1:
        return np.asarray(z, float)
    n = len(z)
    c = np.concatenate(([0.0], np.cumsum(z, dtype=np.float64)))
    i = np.arange(n)
    lo = np.clip(i - k//2, 0, n)
    hi = np.clip(i - k//2 + k, 0, n)
    return (c[hi] - c[lo])/(hi - lo)

def _close(m, gap=3):
    """A mask with its holes of up to `gap` points filled (where widths' detections meet they can leave a bin out)"""
    s, e = runs(~m)
    inner = (s > 0) & (e < len(m)) & (e - s <= gap)
    m = m.copy()
    for a, b in zip(s[inner], e[inner]):
        m[a:b] = True
    return m

def _paint(n, starts, stops, values=None):
    """A mask (and the most of `values` per bin) over n bins from spans that may overlap"""
    starts, stops = np.clip(starts, 0, n), np.clip(stops, 0, n)
    d = np.bincount(starts, minlength=n + 1) - np.bincount(stops, minlength=n + 1)
    m = np.cumsum(d[:n]) > 0
    if values is None:
        return m, None
    v = np.full(n, -np.inf, np.float32)
    lens = np.maximum(stops - starts, 0)
    if lens.sum():
        idx = np.repeat(starts - np.concatenate(([0], np.cumsum(lens)[:-1])), lens) + np.arange(lens.sum())
        np.maximum.at(v, idx, np.repeat(np.asarray(values, np.float32), lens))
    return m, v

# --- settings by effort

_SETTINGS = {
    "fast": dict(overlap=0.0, max_scale=1/4, groups=(8,)),
    "balanced": dict(overlap=0.5, max_scale=1/4, groups=(2, 8)),
    "max": dict(overlap=0.5, max_scale=1/4, groups=(2, 8)),
}

# --- the steps

def spectrogram(x, Fs, nfft, overlap=0.5):
    """P[t, f] (fftshifted, power per Hz) and the hop"""
    x = np.ascontiguousarray(x, dtype=np.complex64)
    hop = max(1, int(nfft*(1 - overlap)))
    M = (len(x) - nfft)//hop + 1
    if M < 1:
        raise ValueError(f"Need at least {nfft} samples")
    win = _hann_win(nfft)
    frames = np.lib.stride_tricks.as_strided(x, (M, nfft), (x.strides[0]*hop, x.strides[0]))
    X = np.fft.fft(frames*win, axis=1) # (in double: numpy's complex64 FFT is the slower)
    P = X.real**2
    P += X.imag**2
    P = np.fft.fftshift(P, axes=1).astype(np.float32)
    P *= np.float32(1/(np.sum(win**2)*Fs))
    return P, hop

@functools.lru_cache(maxsize=16)
def _hann_win(nfft):
    """The Hann window, float32 values held in double"""
    return np.hanning(nfft).astype(np.float32).astype(np.float64)

def _run_min(v, w):
    """Minimum over each window of 2w+1 centred on each point (the ends held at their value): van Herk / Gil-Werman, the
    minima of blocks of the window's length from each side, O(n) whatever the window"""
    n, W = len(v), 2*w + 1
    pad = np.concatenate((np.full(w, v[0]), v, np.full(w + (-(n + 2*w)) % W, v[-1])))
    blk = pad.reshape(-1, W)
    pre = np.minimum.accumulate(blk, axis=1).ravel()
    suf = np.minimum.accumulate(blk[:, ::-1], axis=1)[:, ::-1].ravel()
    i = np.arange(n)
    return np.minimum(suf[i], pre[i + W - 1])

class _Floor:
    """The noise floor per bin, fitted block by block over the usable bins a..b (the module's step 2)"""

    def __init__(self, A, a, b, nblocks):
        self.A, self.a, self.b = A, a, b
        self.nb = int(max(2, min(nblocks, (b - a)//8)))
        self.edges = a + np.linspace(0, b - a, self.nb + 1).astype(int)
        self.centres = 0.5*(self.edges[:-1] + self.edges[1:] - 1)
        self.bs = int(self.edges[1] - self.edges[0])

    def _spread(self, lv):
        """Block levels (dB, NaN: unknown) to a floor per bin: interpolated between block centres, constant past the
        first and last"""
        ok = np.isfinite(lv)
        if not ok.any():
            return np.full(len(self.A), np.median(self.A[self.a:self.b]))
        return 10**(np.interp(np.arange(len(self.A)), self.centres[ok], lv[ok])/10)

    def first(self):
        """Each block's 2nd percentile, then a line through the blocks fitted again without those
        standing 1.5 dB over it (FCME with a slope: a tilt stays in the floor, a plateau even wider than half the band
        drops out; a curve would bend over a plateau in the middle of the band. The passband's curvature and ripple are
        the block fit's to follow)"""
        # (unsmoothed, a low percentile: in a row of carriers side by side the floor shows only in the notches between
        # them, a bin or two wide, which any smoothing fills; the refit from the noise bins takes out the low bias)
        s = 10*np.log10(self.A + 1e-30)
        lv = np.percentile(self._blocks(s), 2, axis=1)
        # (kept: a block with no noise left to fit is held to its notches, the least of its 4-bin means: a notch is a few
        # bins of ~1000, under any percentile; the offset the noise blocks show takes the bias out)
        self.low = np.min(self._blocks(10*np.log10(_box(self.A, 4) + 1e-30)), axis=1)
        u = np.linspace(-1, 1, self.nb)
        keep = np.ones(self.nb, bool)
        for _ in range(12):
            fit = np.polyval(np.polyfit(u[keep], lv[keep], 1 if keep.sum() >= 4 else 0), u)
            new = lv - fit < 1.5
            if new.sum() < max(3, self.nb//5): # (at least the lowest fifth)
                new = lv <= np.sort(lv - fit)[max(3, self.nb//5) - 1] + fit
            if (new == keep).all():
                break
            keep = new
        return self._spread(np.where(keep, lv, fit))

    def _blocks(self, v):
        """v's blocks as rows (each block's first `bs` bins: the blocks differ by a bin at most)"""
        return v[self.edges[:-1, None] + np.arange(self.bs)[None, :]]

    def fit(self, noise):
        """Each block's mean over the bins marked noise (NaN where under a fifth of it is)"""
        m = self._blocks(noise)
        cnt = m.sum(axis=1)
        tot = (self._blocks(self.A)*m).sum(axis=1)
        ok = cnt >= max(4, 0.2*self.bs)
        lv = np.where(ok, 10*np.log10(np.where(ok, tot/np.maximum(cnt, 1), 1.0) + 1e-30), np.nan)
        # A block with too little noise left (a row of carriers fills it): its neighbours' level, or its notches' (the
        # floor shows between carriers side by side) when lower, at the offset the blocks with noise show between the two
        if ok.any() and getattr(self, "low", None) is not None:
            off = float(np.median(lv[ok] - self.low[ok]))
            notch = self.low + off
            if (~ok).any():
                interp = np.interp(self.centres, self.centres[ok], lv[ok])
                lv = np.where(ok, lv, np.minimum(interp, notch))
            lv = np.where(ok & (notch < lv - 2.0), notch, lv)
        return self._spread(lv)

def _rolloff(A, cut, bs):
    """The usable bins a..b: past the receiver's roll-off at each end, where the spectrum's lower envelope (its least
    over a block: the floor between carriers, which carriers don't lift) is more than 1.5 dB under that envelope's level a
    few blocks in"""
    n = len(A)
    s = 10*np.log10(_box(A, max(1, bs//4)) + 1e-30)
    env = _run_min(s, max(1, bs//2))
    a, b = cut, n - cut
    span = max(4*bs, 8)
    if a + span < b:
        ref = np.median(env[a + bs:a + span])
        up = np.flatnonzero(env[a:a + span] >= ref - 1.5)
        a = a + (int(up[0]) if len(up) else 0)
    if b - span > a:
        ref = np.median(env[b - span:b - bs])
        up = np.flatnonzero(env[b - span:b][::-1] >= ref - 1.5)
        b = b - (int(up[0]) if len(up) else 0)
    return a, b

class _Detector:
    """Thresholds and detection on a whitened spectrum (the module's step 3)"""

    def __init__(self, nfft, hop, M, pfa, kmax, n_groups=0, sys_db=0.0, p_lo=1e-2):
        self.nfft, self.hop, self.M = nfft, hop, M
        self.levels = int(np.log2(max(1, kmax))) + 1
        tests = sum(2*nfft/(1 << L) for L in range(self.levels))*(1 + n_groups)
        self.p_hi = pfa/max(tests, 1.0)
        self.p_lo = p_lo
        self.sys = sys_db/4.343 # natural-log spread of the floor's own error
        self._t = {}

    def thresholds(self, k, m):
        key = (k, m)
        t = self._t.get(key)
        if t is None:
            t = self._t[key] = self._thresholds(k, m)
        return t

    def _thresholds(self, k, m):
        n = _dof(self.nfft, self.hop, m, k)
        hi = _gamma_isf(self.p_hi, n)*np.exp(_norm_isf(self.p_hi)*self.sys)
        lo = _gamma_isf(self.p_lo, n)*np.exp(_norm_isf(self.p_lo)*self.sys)
        return float(hi), float(lo)

    POOL, POOL_FROM = 16, 64 # widths of POOL_FROM bins and more are tested on means of POOL bins

    def detect(self, Z, m, local=1.0):
        """Occupied bins of the whitened spectrum Z (a mean of m frames; its noise at `local`): at each width k = 2^L,
        finest first, the means of k bins at a stride of k/2, seeded over the high threshold and grown over the low one;
        a run of windows marks the bins from its first window's centre - k/4 to its last's + k/4. What a width marks is
        set back to the noise (excised, as CLEAN does) before the next: a wider width finds only what the narrower ones
        could not, not a strong narrow signal again with its surroundings.

        A 2-D Z is rows (groups of frames) tested together: each width's windows over every row, and the most of them
        over the rows tested (the thresholds count every row's windows); what it marks is excised from every row.
        Windows of k bins are pairs of blocks of k/2 (each width's blocks pairs of the last's; an excision sums again
        only the blocks it touched), and from POOL_FROM bins on everything runs on means of POOL bins: their windows and
        marks fall on multiples of POOL bins, so this is exact, at a POOL-th of the work.
        Returns (mask, score: dB over the high threshold, the most over widths; the narrowest width that marked each
        bin), each over Z's bins. The C++ kernel when built (rpps._native.find_detect), else _detect_py, the reference
        it mirrors"""
        from ... import _accel
        if _accel.native is None:
            return self._detect_py(Z, m, local)
        Zx = np.array(np.atleast_2d(Z), np.float64)
        Zx /= local
        n = Zx.shape[1]
        ks = [1 << L for L in range(self.levels) if (1 << L) <= n]
        th = np.array([self.thresholds(k, m) for k in ks], np.float64).reshape(-1, 2)
        return _accel.native.find_detect(Zx, np.ascontiguousarray(th[:, 0]), np.ascontiguousarray(th[:, 1]),
                                         self.POOL, self.POOL_FROM)

    def _detect_py(self, Z, m, local=1.0):
        """detect's loop in numpy (the reference the kernel is tested against)"""
        Zx = np.array(np.atleast_2d(Z), np.float64)
        Zx /= local
        R, n = Zx.shape
        top = 1 << (self.levels - 1)
        while top > n:
            top >>= 1
        n_pad = -(-n//max(top, self.POOL))*max(top, self.POOL) # (blocks of every width fit; the padding is noise)
        if n_pad > n:
            Zx = np.concatenate((Zx, np.ones((R, n_pad - n))), axis=1)
        res = [] # (resolution, occ, score, kmin) of the bin and pooled stages
        pool = 1
        occ, score, kmin = np.zeros(n_pad, bool), np.full(n_pad, -np.inf, np.float32), np.full(n_pad, np.inf)
        B = None # block sums of Zx, blocks of h = k/2
        for L in range(self.levels):
            k = 1 << L
            if k > n:
                break
            if pool == 1 and k >= self.POOL_FROM:
                res.append((1, occ, score, kmin))
                pool = self.POOL
                Zx = Zx.reshape(R, n_pad//pool, pool).mean(axis=2)
                occ, score, kmin = np.zeros(n_pad//pool, bool), np.full(n_pad//pool, -np.inf, np.float32), np.full(n_pad//pool, np.inf)
                B = None
            ncur = Zx.shape[1]
            kw = k//pool
            n_win = n - k + 1 if k == 1 else (n - k)//(k//2) + 1 # windows inside the real bins
            if k == 1:
                Y = Zx[:, :n_win]
            else:
                h = kw//2
                if B is None:
                    B = Zx.reshape(R, ncur//h, h).sum(axis=2)
                elif B.shape[1] != ncur//h:
                    B = B.reshape(R, -1, 2).sum(axis=2)
                Y = (B[:, :n_win] + B[:, 1:n_win + 1])/kw
            if R > 1:
                Y = Y.max(axis=0)
            else:
                Y = Y[0]
            hi, lo = self.thresholds(k, m)
            seed = Y > hi
            if not seed.any():
                continue
            gs, ge = runs(Y > lo)
            cs = np.concatenate(([0], np.cumsum(seed)))
            keep = cs[ge] > cs[gs]
            gs, ge = gs[keep], ge[keep]
            # Each run's most: reduceat runs on to the next run's start, but the gap between is under the low
            # threshold and the run reaches over the high one
            best = 10*np.log10(np.maximum(np.maximum.reduceat(Y, gs), 1e-30)/hi).astype(np.float32)
            if k == 1:
                a0, b0 = gs, ge
            else: # (in bins, then in this resolution's units: multiples of POOL from POOL_FROM on)
                a0 = (gs*(k//2) + k//4)//pool
                b0 = ((ge - 1)*(k//2) + k - k//4)//pool
            m_k, v_k = _paint(ncur, a0, b0, best)
            occ |= m_k
            np.maximum(score, v_k, out=score)
            kmin[m_k] = np.minimum(kmin[m_k], k)
            Zx[:, m_k] = 1.0
            if k > 1: # the blocks the excision touched, summed again
                bc = np.unique(np.flatnonzero(m_k)//h)
                B[:, bc] = Zx.reshape(R, ncur//h, h)[:, bc].sum(axis=2)
            else:
                B = None # (the first blocks are taken from Zx at width 2)
        res.append((pool, occ, score, kmin))
        out_o, out_s, out_k = np.zeros(n_pad, bool), np.full(n_pad, -np.inf, np.float32), np.full(n_pad, np.inf)
        for pool, o, sc, km in res:
            if pool > 1:
                o, sc, km = (np.repeat(v, pool) for v in (o, sc, km))
            out_o |= o
            np.maximum(out_s, sc, out=out_s)
            np.minimum(out_k, km, out=out_k)
        return out_o[:n], out_s[:n], out_k[:n]

    def passes(self, z, m, local=1.0):
        """Whether the mean of these bins (a box: its own width) is over the high threshold"""
        hi, _ = self.thresholds(len(z), m)
        return float(np.mean(z))/local > hi

def _nfft_for(N):
    return 1 << int(np.clip(np.floor(np.log2(max(N//16, 64))), 6, 16))

class _Ctx:
    """What every span's measurement needs"""

    def band(self, a, b):
        """Each frame's mean over bins a..b of the whitened frames (from their running sums over bins, Cw: a Finder
        keeps them frame by frame; made here once otherwise)"""
        if getattr(self, "Cw", None) is None:
            self.Cw = _cumbins(self.Pw)
        return (self.Cw[:, b] - self.Cw[:, a])/max(b - a, 1)

def _cumbins(Pw):
    """Each frame's running sum over its bins, from 0 (float64: (M, nfft + 1))"""
    C = np.zeros((len(Pw), Pw.shape[1] + 1))
    np.cumsum(Pw, axis=1, dtype=np.float64, out=C[:, 1:])
    return C

def find(x, Fs: float = 1.0, cf=0.0, nfft=None, pfa=1e-3, edge=0.005, drop_db=10.0, min_bins=1, min_across=8,
         blocks=64, sys_db=None):
    """Signals in a capture (see the module): a Survey of Signal sorted by frequency, each with t0/t1 (when it is on,
    [s from the start]), bursts ([(t0, t1)], None when continuous or too weak a frame to tell; one with bursts is a
    Burst), score (dB over the threshold that found it) and over (the box it stands on, or None)

    pfa: expected false boxes in a capture of pure noise (spread over every width and group tested)
    nfft: FFT size [N/16, 64 to 65536]
    edge: fraction of the band at each end not looked at [0.5%: the receiver's roll-off is found from the spectrum, and
        carriers can run to within 1% of the band's edge]
    drop_db: a carrier's edges are this far under its top [dB] (or halfway to the floor or valley beside it)
    min_bins: narrowest signal; min_across: bins across under which its edges aren't trusted (Signal.resolved False)
    blocks: blocks the floor is fitted in (at least 8 bins each); sys_db: the floor's own error [measured]
    For a stream (a display), Finder: the same, block by block.
    """
    st = _SETTINGS[_effort.level()]
    x = np.asarray(x)
    nfft = _nfft_for(len(x)) if nfft is None else int(nfft)
    P, hop = spectrogram(x, Fs, nfft, st["overlap"])
    fit = fit_floor(P, hop, edge, blocks, sys_db)
    return _find_frames(P, hop, x, Fs, cf, fit, pfa, edge, drop_db, min_bins, min_across)

class FloorFit:
    """The noise floor fitted to a spectrogram (find's step 2): floor per bin (power per Hz), the usable bins a..b (past
    the receiver's roll-off), the floor's own error sys (natural log) and busy: the bins clearly signal (left out of the
    fit, before it was widened)"""
    __slots__ = ("floor", "a", "b", "sys", "busy")

    def __init__(self, floor, a, b, sys, busy):
        self.floor, self.a, self.b, self.sys, self.busy = floor, a, b, sys, busy

def fit_floor(P, hop, edge=0.005, blocks=64, sys_db=None):
    """find's step 2 on a spectrogram P[t, f]: the FloorFit"""
    M, nfft = P.shape
    A = P.mean(axis=0, dtype=np.float64)
    cut = int(np.ceil(max(edge, 0.0)*nfft))
    fl = _Floor(A, cut, nfft - cut, blocks)
    a_use, b_use = _rolloff(A, cut, fl.bs)
    fl = _Floor(A, a_use, b_use, blocks)
    floor = fl.first()
    # Only what is clearly signal is cut out of the fit: narrow lines and carriers (the finest widths), and levels
    # 1.5 dB over the floor over an eighth of a block (0.1 dB of scatter). What the coarse widths find at a fraction of
    # a dB is as likely the floor's own misfit, which cutting out would freeze (the fit can't correct what it can't see)
    fine = _Detector(nfft, hop, M, 1e-2, 16)
    for _ in range(3):
        Z = A/floor
        occ, _s, _k = fine.detect(Z, M)
        occ |= 10*np.log10(_box(Z, max(1, fl.bs//8)) + 1e-30) > 1.5
        busy = occ
        # Widen each cut with its signal: the floor beside a wide hump is fitted past its flanks, not up them
        s0, s1 = runs(occ)
        grow = np.clip((s1 - s0)//2, fl.bs//8, 2*fl.bs) # (at most two blocks: past that the passband's ripple misleads)
        occ = _paint(nfft, s0 - grow, s1 + grow)[0]
        lone = Z > _gamma_isf(1e-3, _dof(nfft, hop, M, 1)) # lines a box mean leaves
        floor = fl.fit(~occ & ~lone)
    sys = (sys_db or 0.0)/4.343
    if sys_db is None: # what the noise bins scatter beyond chance, over a quarter block
        Z = A/floor
        noise = ~occ & ~lone
        noise[:a_use] = noise[b_use:] = False
        kr = max(1, fl.bs//4)
        r = 10*np.log10(_box(Z, kr) + 1e-30)[noise]
        if len(r) > 8*kr:
            emp = 1.4826*np.median(np.abs(r - np.median(r)))
            sys = np.sqrt(max(0.0, emp**2 - (4.343/np.sqrt(_dof(nfft, hop, M, kr)))**2))/4.343
    return FloorFit(floor, a_use, b_use, float(sys), busy)

def _find_frames(P, hop, x, Fs, cf, fit, pfa, edge, drop_db, min_bins, min_across, t_start=0.0, state=None):
    """find's steps 3-6 on a spectrogram P[t, f] (frames at hop) of the samples x (None: bursts timed by the frames),
    with the floor `fit`

    state: a Finder's, carried from step to step (None: nothing kept): its detector, mean spectrum, whitened frames
    and their running sums over bins, the group detections of groups already complete (frames counted from the
    stream's start, `frame0` the first held), and each region's reading (levels.boxes)"""
    st = _SETTINGS[_effort.level()]
    M, nfft = P.shape
    floor = fit.floor
    bin_hz = Fs/nfft
    freq = (np.arange(nfft) - nfft//2)*bin_hz
    cut = int(np.ceil(max(edge, 0.0)*nfft))
    kmax = max(1, int(st["max_scale"]*nfft))
    groups = [g for g in st["groups"] if g <= M//2]
    n_groups = sum(M//g for g in groups)
    if state is None:
        A = P.mean(axis=0, dtype=np.float64)
        Pw = P/floor[None, :].astype(np.float32)
        det = _Detector(nfft, hop, M, pfa, kmax, n_groups)
        det.sys = fit.sys
    else:
        A, Pw = state["A"], state["Pw"]
        det = state.get("det")
        if det is None or det.M != M:
            det = state["det"] = _Detector(nfft, hop, M, pfa, kmax, n_groups)
            det.sys = fit.sys
            state["groups"] = {}
    Z = A/floor

    # Step 3: occupied bins, on the mean and on each group of frames
    occ, score, kmin = det.detect(Z, M)
    for g in groups:
        if state is None: # every group of g frames at once
            o, sg, kg = det.detect(Pw[:M//g*g].reshape(M//g, g, nfft).mean(axis=1), g)
        else: # groups at fixed frames of the stream, each detected once, when complete
            f0 = state["frame0"]
            o, sg, kg = np.zeros(nfft, bool), np.full(nfft, -np.inf, np.float32), np.full(nfft, np.inf)
            for j in range(-(-f0//g), (f0 + M)//g):
                r = state["groups"].get((g, j))
                if r is None:
                    r = state["groups"][(g, j)] = det.detect(Pw[j*g - f0:(j + 1)*g - f0].mean(axis=0), g)
                o |= r[0]
                np.maximum(sg, r[1], out=sg)
                np.minimum(kg, r[2], out=kg)
            state["groups"] = {key: v for key, v in state["groups"].items() if key[1]*key[0] + key[0] > f0}
        occ |= o
        np.maximum(score, sg, out=score)
        np.minimum(kmin, kg, out=kmin)
    occ = _close(occ)
    occ[:cut] = occ[nfft - cut:] = False

    c = _Ctx()
    T = ((M - 1)*hop + nfft)/Fs
    c.__dict__.update(x=x, fine={}, Z=Z, A=A, Pw=Pw, Cw=state.get("Cw") if state is not None else None, floor=floor, det=det, nfft=nfft, hop=hop, M=M, Fs=Fs, cf=cf,
                      bin_hz=bin_hz, freq=freq, cut=(fit.a, fit.b), drop_db=drop_db,
                      min_bins=min_bins, min_across=min_across, T=T, score=score, kmin=kmin)
    out = levels.boxes(c, occ, groups, state=state) # (steps 4-6: rpps.analysis.spectrum.levels)
    if t_start:
        for s_ in out:
            s_.t0, s_.t1 = s_.t0 + t_start, s_.t1 + t_start
            if s_.bursts:
                s_.bursts = [(u + t_start, v + t_start) for u, v in s_.bursts]
    # (a box half inside another, not standing on it: a skirt measured from a span of its own)
    if len(out) > 1:
        lo, hi = np.array([s.lo for s in out]), np.array([s.hi for s in out])
        w, holds = np.array([s.width for s in out]), np.array([bool(s.holds) for s in out])
        half = (np.minimum(hi[:, None], hi[None, :]) - np.maximum(lo[:, None], lo[None, :]) >= 0.5*w[:, None]) & \
            (w[None, :] > w[:, None]) & ~holds[None, :]
        out = [s for s, h in zip(out, half.any(axis=1)) if s.over is not None or not h]
    for s in out: # (a signal on part of the time is a Burst; one kept from a step before may be on all of it now)
        s.__class__ = Burst if s.bursts else Signal
    out.sort(key=lambda s: s.mid)
    for i, s in enumerate(out):
        s.id = i
    sv = Survey(out, Fs, cf, nfft, M, freq, 10*np.log10(A + 1e-30), float(np.median(10*np.log10(floor[fit.a:fit.b]))))
    sv.floor_db = 10*np.log10(floor)
    return sv

class _Tail:
    """The newest `n` rows of a stream of rows (frames, or samples), contiguous and in order, without copying all of
    them at each push: a buffer of twice that, its rows moved back to the start when it fills"""

    def __init__(self, n, shape=(), dtype=np.float32):
        self.n = int(n)
        self.buf = np.zeros((2*self.n,) + tuple(shape), dtype)
        self.i0 = self.i1 = 0

    def __len__(self):
        return self.i1 - self.i0

    @property
    def view(self):
        return self.buf[self.i0:self.i1]

    def push(self, rows):
        rows = rows[-self.n:]
        k = len(rows)
        if self.i1 + k > len(self.buf): # (full: what is kept moved to the start)
            keep = min(len(self), self.n - k)
            self.buf[:keep] = self.buf[self.i1 - keep:self.i1]
            self.i0, self.i1 = 0, keep
        self.buf[self.i1:self.i1 + k] = rows
        self.i1 += k
        self.i0 = max(self.i0, self.i1 - self.n)

    def drop(self, k):
        """Forget the oldest k rows"""
        self.i0 = min(self.i1, self.i0 + int(k))

class Finder:
    """Signals in a stream, block by block (a live display): find over the last `frames` frames, the floor refitted every
    `refit` frames [4 x frames: the receiver's passband changes slowly], ids kept from block to block (Survey.match)

        f = Finder(Fs, nfft=4096, cf=cf, frames=frames_in_a_second)
        for blk in source(65536):
            sv = f.step(blk.data)       # a Survey (None before a first frame); times [s] from the stream's start

    For a display, hold about a second of frames: what is shown then changes little from step to step, and a box
    follows a change within about that. A step costs the FFTs of its new frames, a detection on the mean (a running
    sum) and on each new group of frames (one complete is detected once), and the boxes: each region keeps its reading
    while its span and level hold, its boxes measured again, and is read again when they move or once a window
    (rpps.analysis.spectrum.levels.boxes; track=False: every step). Bursts are timed by the frames (find times them
    finer). Frames are cut at fixed places in the stream and the floor refitted at fixed frame counts, so without
    tracking what it holds after a stretch of samples doesn't depend on how the stretch came in blocks. overlap: of
    the frames [the effort level's when made]; the rest (pfa, edge, drop_db, ...) as find's.

    hold [s]: how long a signal not found again stays in the Survey, marked stale (Signal.stale, as last seen;
    Signal.seen when it was last found): -1 for good [the default: a display keeps what it has seen], 0 not at all. A
    stale signal found again has its id back (Survey.match)"""

    def __init__(self, Fs, nfft=4096, cf=0.0, frames=32, refit=None, overlap=None, track=True, hold=-1.0, **kw):
        self.Fs, self.nfft, self.cf = float(Fs), int(nfft), cf
        self.frames = int(frames)
        self.refit = max(1, int(4*self.frames if refit is None else refit))
        self.track = track
        ov = _SETTINGS[_effort.level()]["overlap"] if overlap is None else overlap
        self.hop = max(1, int(self.nfft*(1 - ov)))
        self.kw = dict(pfa=1e-3, edge=0.005, drop_db=10.0, min_bins=1, min_across=8)
        self.kw.update({k: v for k, v in kw.items() if k in self.kw})
        self.blocks, self.sys_db = kw.get("blocks", 64), kw.get("sys_db")
        self.hold = float(hold)
        self.reset()

    def reset(self):
        """Forget the stream: the next step starts a new one"""
        self._raw = np.zeros(0, np.complex64) # samples from the next frame's start on
        self._P = _Tail(self.frames, (self.nfft,))
        self._Pw = _Tail(self.frames, (self.nfft,)) # the held frames over the floor
        self._Cw = _Tail(self.frames, (self.nfft + 1,), np.float64) # their running sums over bins (_Ctx.band)
        self._sum = np.zeros(self.nfft)       # their sum (the mean spectrum, running)
        self._n = 0                           # frames made so far
        self._fit = None
        self._fit_held = 0                    # frames held at the floor's last fit
        self._state = {}
        self.survey = None

    def _refit_now(self):
        P = self._P.view
        self._fit = fit_floor(P, self.hop, self.kw["edge"], self.blocks, self.sys_db)
        self._fit_held = len(P)
        self._Pw = _Tail(self.frames, (self.nfft,))
        self._Pw.push(P/self._fit.floor[None, :].astype(np.float32))
        self._Cw = _Tail(self.frames, (self.nfft + 1,), np.float64)
        self._Cw.push(_cumbins(self._Pw.view))
        self._sum = P.sum(axis=0, dtype=np.float64) # (exact again: no drift from the running sum)
        self._state = {k: v for k, v in self._state.items() if k in ("regions", "levels")} # (the regions' readings carry over)

    def _add(self, P):
        """Hold these frames (the newest `frames`), refitting the floor at each multiple of `refit` frames made, and
        until enough frames are held"""
        while len(P):
            to = self.refit - self._n % self.refit # frames to the next refit
            take, P = P[:to], P[to:]
            if len(take) >= self.frames:
                self._sum = take[-self.frames:].sum(axis=0, dtype=np.float64)
            else:
                drop = max(0, len(self._P) + len(take) - self.frames)
                self._sum += take.sum(axis=0, dtype=np.float64) - self._P.view[:drop].sum(axis=0, dtype=np.float64)
            self._P.push(take)
            if self._fit is not None:
                w = take/self._fit.floor[None, :].astype(np.float32)
                self._Pw.push(w)
                self._Cw.push(_cumbins(w[-self.frames:]))
            self._n += len(take)
            held = len(self._P)
            if self._n % self.refit == 0 or self._fit is None or \
                    (held < self.frames and held >= 2*self._fit_held) or (held == self.frames > self._fit_held):
                if held >= 2:
                    self._refit_now()

    def step(self, block):
        """Take a block of samples; the Survey of the frames now held (None while there are fewer than 2)"""
        x = np.asarray(block, np.complex64)
        raw = np.concatenate((self._raw, x))
        if len(raw) >= self.nfft:
            P, _ = spectrogram(raw, self.Fs, self.nfft, 1 - self.hop/self.nfft)
            self._add(P)
            raw = raw[len(P)*self.hop:]
        self._raw = raw
        if len(self._P) < 2 or self._fit is None:
            return None
        M = len(self._P)
        st = self._state
        st["A"], st["Pw"], st["Cw"], st["frame0"] = self._sum/M, self._Pw.view, self._Cw.view, self._n - M
        st["frame"], st["refresh"] = self._n, (self.frames if self.track else None)
        t_start = (self._n - M)*self.hop/self.Fs # (the oldest held frame's start)
        sv = _find_frames(self._P.view, self.hop, None, self.Fs, self.cf, self._fit, t_start=t_start, state=st,
                          **self.kw)
        now = self._n*self.hop/self.Fs # (the end of the newest frame [s from the stream's start])
        for g in sv:
            g.stale, g.seen = False, now
        if self.survey is not None:
            sv.match(self.survey)
            if self.hold != 0: # (what was not found again: kept, stale, for its hold)
                found = {g.id for g in sv}
                kept = [g for g in self.survey if g.id not in found and (self.hold < 0 or now - g.seen <= self.hold)]
                for g in kept:
                    g.stale = True
                sv.extend(kept)
                sv.sort(key=lambda g: g.mid)
        self.survey = sv
        return sv

# --- a Finder's regions from step to step

TRACK_DB = 0.5 # a region's level moving more than this since it was read: read again

def _same(recs, a, b, lvl):
    """The region of the last step that a..b (level lvl dB) is still: its bins to 2% of its width, its level to
    TRACK_DB; None"""
    tol = max(2, int(0.02*(b - a)))
    for r in recs:
        if abs(a - r["a"]) <= tol and abs(b - r["b"]) <= tol and abs(lvl - r["lvl"]) <= TRACK_DB:
            return r
    return None

def _remeasure(c, sigs):
    """Copies of a region's Signals measured again on this step's spectrum (a Finder's region that has not moved): SNR
    (as read, moved by the change of the power over the floor across its reach, Signal._range), power, and the bursts
    of one that had them (at the frames' resolution: the finer times come with a reading)"""
    out = _copies(sigs)
    for g in out:
        if g is None:
            continue
        x0, x1, i0, i1, loc = g._range
        excess = float(np.sum(c.Z[x0:x1]/loc - 1.0))
        if excess > 0 and g._excess > 0:
            g.snr = float(g._snr0 + 10*np.log10(excess/g._excess))
        g.power = float(10*np.log10(c.A[i0:i1].sum()*c.bin_hz + 1e-30))
        g.t0, g.t1 = 0.0, c.T
        if g.bursts:
            on, g.bursts, g.duty = _on(c, i0, i1, loc, fine=False)
            if g.bursts:
                g.t0, g.t1 = g.bursts[0][0], g.bursts[-1][1]
    return out

def _copies(sigs):
    """Copies of Signals (None stays None), each one's `over` pointing at the copy of the one it stood on"""
    import copy
    new = {id(s): copy.copy(s) for s in sigs if s is not None}
    for s in new.values():
        if s.over is not None:
            s.over = new.get(id(s.over), s.over)
    return [None if s is None else new[id(s)] for s in sigs]

def _on(c, a, b, local, fine=True):
    """Step 4: the frames a span is on (None: continuous, or too weak a frame to tell), its bursts [(t0, t1)] and duty.
    The frames say which to average; the bursts' times come from frames r times shorter (r as large as leaves the span
    32 bins wide: about 16 degrees of freedom a frame), gated the same way with one-frame gaps closed, when the span stands
    out of the noise in them too (fine=False, or no samples in c (a Finder): the frames' times)"""
    hi, lo = c.det.thresholds(b - a, 1)
    band = c.band(a, b)/_mean(local, a, b)
    if band.mean() < 2*hi: # a frame alone can't tell it from noise
        return None, None, 1.0
    on = _gate(band, hi, lo)
    if on.all() or not on.any():
        return None, None, 1.0
    T = c.T
    bursts = [(p*c.hop/c.Fs, min(T, ((q - 1)*c.hop + c.nfft)/c.Fs)) for p, q in zip(*runs(on))]
    duty = float(on.mean())
    r = 1 << int(np.clip(np.floor(np.log2(max((b - a)/32, 1))), 0, np.log2(max(c.nfft//16, 1))))
    if r > 1 and fine and c.x is not None:
        nt = c.nfft//r
        if nt not in c.fine:
            Pt, _ = spectrogram(c.x, c.Fs, nt, 0.0)
            ft = c.floor[:c.nfft//r*r].reshape(-1, r).mean(axis=1)
            c.fine[nt] = Pt/ft[None, :].astype(np.float32)
        Zt = c.fine[nt]
        ta, tb = a//r, max(a//r + 1, -(-b//r))
        hi_t, lo_t = (_gamma_isf(c.det.p_hi, _dof(nt, nt, 1, tb - ta)), _gamma_isf(c.det.p_lo, _dof(nt, nt, 1, tb - ta)))
        bt = Zt[:, ta:tb].mean(axis=1)/_mean(local, a, b)
        on_t = _gate(bt, hi_t, lo_t)
        gap = np.flatnonzero(~on_t[1:-1] & on_t[:-2] & on_t[2:]) + 1 # (a frame's dip inside a burst)
        on_t[gap] = True
        if bt[on_t].mean() > 2*hi_t if on_t.any() else False:
            bursts = [(p*nt/c.Fs, min(T, q*nt/c.Fs)) for p, q in zip(*runs(on_t))]
            duty = float(on_t.mean())
    return on, bursts, duty

def _gate(band, hi, lo):
    """Frames on: runs over `lo` that reach `hi` (hysteresis)"""
    seed, grow = band > hi, band > lo
    cs = np.concatenate(([0], np.cumsum(seed)))
    on = np.zeros(len(band), bool)
    for p, q in zip(*runs(grow)):
        if cs[q] > cs[p]:
            on[p:q] = True
    return on

def _mean(local, a, b):
    """A local floor's level over bins a..b (local: 1, or a level per bin)"""
    return float(local) if np.isscalar(local) else float(np.mean(local[a:b]))

# --- Steps 5-6: the span's spectrum as a staircase of flat levels (optimal changepoints), its signals read off the levels

