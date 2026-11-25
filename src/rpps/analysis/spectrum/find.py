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
5. The span as a staircase: its spectrum, with context each side, in points of up to 8 bins (dB), fitted with flat
   levels by optimal partitioning (PELT, exact; each point weighted by its own scatter, the noise's or the roughness
   about it, a change costing 3 log n; C++ kernel). Runs of levels over the floor are signals. A run is cut at a notch (a
   level 3/4 x `split_db` under the highest each side) and at a step between two broad levels that far apart (a carrier
   flush against a stronger one). A run whose lowest broad level stands over the floor, covers a third of it and is
   never cut under is a pedestal: listed with `holds`, each group of levels on it read the same way over it. Narrower
   levels standing over a piece's own (its longest) level are a signal on it (a jammer on a carrier): listed with `over`
   (the piece), the piece with `holds`; whether a box with `holds` is a raised floor or a signal under a jammer, the
   spectrum can't say (its own symbol-rate line can). Lines narrower than the points (tones), found over the staircase,
   are boxes of their own when 3 dB over it; 4 or more side by side at one spacing are one box, a comb (`lines`).
6. Each piece's edges: where the spectrum falls `drop_db` under its level, or halfway to the level outside it (the floor,
   a notch, a pedestal) when that is less. A piece whose level never falls on a side facing the floor is part of
   something wider; one whose edge rises over most of its width (1/4 to 3/4 of its height on the staircase, measured
   from outside in, wider than a fifth of it or 2% of the band) is a hump; a piece cut by the band's edge stands over
   the other side by 3/4 x `split_db`; every box passes the detector at its own width.

Effort (rp.set_effort): fast skips frame overlap and tests bursts only in groups of 8 frames; balanced and max overlap
frames by half and test groups of 2 and 8. Every level tests widths up to a quarter of the band: capped
lower, a wide weak signal or a hump is found as fragments, each passing on its own.
"""
import functools

import numpy as np

from ... import _effort
from .signal import Signal, Survey

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

@functools.lru_cache(maxsize=4096)
def _dof_cached(nfft, hop, M, k):
    win = np.hanning(nfft)
    w2 = win**2
    e2 = w2.sum()
    lags = np.arange(1, min(k, 8))
    cb = 1.0
    if len(lags):
        W = np.fft.fft(w2)
        cb = 1 + 2*np.sum((1 - lags/k)*np.abs(W[lags])**2/e2**2)
    ct = 1.0
    lt = np.arange(1, min(M, int(np.ceil(nfft/hop))))
    if len(lt):
        rt = np.array([(win[j*hop:]*win[:nfft - j*hop]).sum()**2/e2**2 for j in lt])
        ct = 1 + 2*np.sum((1 - lt/M)*rt)
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

def _runs(m):
    """(starts, stops) of the True runs of a 1-D mask (stops exclusive)"""
    d = np.diff(np.concatenate(([0], m.view(np.int8) if m.dtype == bool else m.astype(np.int8), [0])))
    return np.flatnonzero(d == 1), np.flatnonzero(d == -1)

def _close(m, gap=3):
    """A mask with its holes of up to `gap` points filled (where widths' detections meet they can leave a bin out)"""
    s, e = _runs(~m)
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
    win = np.hanning(nfft).astype(np.float32)
    frames = np.lib.stride_tricks.as_strided(x, (M, nfft), (x.strides[0]*hop, x.strides[0]))
    X = np.fft.fft(frames*win, axis=1)
    P = X.real**2 + X.imag**2
    P = np.fft.fftshift(P, axes=1).astype(np.float32)
    P *= np.float32(1/(np.sum(win.astype(np.float64)**2)*Fs))
    return P, hop

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
        bin), each over Z's bins"""
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
            gs, ge = _runs(Y > lo)
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

def _walk(v, i, stop, thr, step, g=1):
    """From i, step while v stays above thr (a dip under it shorter than g points is stepped over) and short of `stop`;
    the crossing (fractional index)"""
    n = len(v)
    j = i
    while True:
        for k in range(1, g + 1):
            nj = j + step*k
            if nj == stop or not 0 <= nj < n:
                k = None
                break
            if v[nj] > thr:
                break
        else:
            k = None
        if k is None:
            break
        j += step*k
    nxt = j + step
    if nxt != stop and 0 <= nxt < n and v[j] > v[nxt]:
        return j + step*float(np.clip((v[j] - thr)/(v[j] - v[nxt]), 0, 1))
    return j + 0.5*step

DEBUG = None # a list: find() appends why it dropped each piece it looked at

def _why(c, reason, e0, e1):
    if DEBUG is not None:
        DEBUG.append((reason, (e0 + 0.5 - c.nfft//2)*c.bin_hz, (e1 + 0.5 - c.nfft//2)*c.bin_hz))
    return None

class _Ctx:
    """What every span's measurement needs"""
    pass

def find(x, Fs: float = 1.0, cf=0.0, nfft=None, pfa=1e-3, edge=0.005, split_db=4.0, drop_db=10.0, min_bins=1,
         min_across=8, blocks=64, sys_db=None, segment="stair"):
    """Signals in a capture (see the module): a Survey of Signal sorted by frequency, each with t0/t1 (when it is on,
    [s from the start]), bursts ([(t0, t1)], None when continuous or too weak a frame to tell), score (dB over the
    threshold that found it), over (the box it stands on, or None), and its shape: edge (its edges' 25-75% transition,
    the wider, over its width: a raised-cosine carrier's is under beta/3, a hump's most of it) and rough (its top's
    roughness beyond the noise's, dB: spectral lines, a rough hump)

    pfa: expected false boxes in a capture of pure noise (spread over every width and group tested)
    nfft: FFT size [N/16, 64 to 65536]
    edge: fraction of the band at each end not looked at [0.5%: the receiver's roll-off is found from the spectrum, and
        carriers can run to within 1% of the band's edge]
    split_db: least valley between carriers side by side [dB; 4x the spectrum's ripple when more]
    drop_db: a carrier's edges are this far under its top [dB] (or halfway to the floor or valley beside it)
    min_bins: narrowest signal; min_across: bins across under which its edges aren't trusted (Signal.resolved False)
    blocks: blocks the floor is fitted in (at least 8 bins each); sys_db: the floor's own error [measured]
    segment: how steps 5-6 read what the detector marked: "stair" (the staircase and its rules) or "net" (the learned
        segmenter, rpps.analysis.spectrum.net)
    For a stream (a display), Finder: the same, block by block.
    """
    st = _SETTINGS[_effort.level()]
    x = np.asarray(x)
    nfft = _nfft_for(len(x)) if nfft is None else int(nfft)
    P, hop = spectrogram(x, Fs, nfft, st["overlap"])
    fit = fit_floor(P, hop, edge, blocks, sys_db)
    return _find_frames(P, hop, x, Fs, cf, fit, pfa, edge, split_db, drop_db, min_bins, min_across, segment=segment)

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
        s0, s1 = _runs(occ)
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

def _find_frames(P, hop, x, Fs, cf, fit, pfa, edge, split_db, drop_db, min_bins, min_across, t_start=0.0, state=None,
                 segment="stair"):
    """find's steps 3-6 on a spectrogram P[t, f] (frames at hop) of the samples x, with the floor `fit`

    state: a Finder's, carried from step to step (None: nothing kept): its detector, mean spectrum, whitened frames,
    the group detections of groups already complete (frames counted from the stream's start, `frame0` the first
    held), and the spans of the last step with their boxes, kept where a span has not moved"""
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
    c.__dict__.update(x=x, fine={}, Z=Z, A=A, Pw=Pw, floor=floor, det=det, nfft=nfft, hop=hop, M=M, Fs=Fs, cf=cf,
                      bin_hz=bin_hz, freq=freq, cut=(fit.a, fit.b), split_db=split_db, drop_db=drop_db,
                      min_bins=min_bins, min_across=min_across, T=T, score=score, kmin=kmin)
    prev = state.get("spans") if state is not None else None
    spans = []
    out = _net(c, occ, groups, t_start) if segment == "net" else []
    for a, b in (zip(*_runs(occ)) if segment != "net" else ()):
        if b - a < min_bins:
            continue
        a, b = int(a), int(b)
        lvl = 10*np.log10(max(float(Z[a:b].mean()), 1e-30))
        got = _tracked(prev, a, b, lvl, c, t_start) if prev else None
        if got is None:
            got = _span_seg(c, a, b)
            for s_ in got:
                if t_start:
                    s_.t0, s_.t1 = s_.t0 + t_start, s_.t1 + t_start
                    if s_.bursts:
                        s_.bursts = [(u + t_start, v + t_start) for u, v in s_.bursts]
        spans.append((a, b, lvl, got))
        out += got
    if state is not None:
        state["spans"] = spans
    out = _own_lines(out) # (a carrier's own comb of lines: its, not signals)
    out = _combs(c, out) # (lines at about one spacing: one box, the comb)
    # (a box half inside another, not standing on it: a skirt measured from a span of its own)
    out = [s for s in out if s.over is not None or not any(
        t is not s and not t.holds and min(s.hi, t.hi) - max(s.lo, t.lo) >= 0.5*s.width and t.width > s.width
        for t in out)]
    out.sort(key=lambda s: s.mid)
    for i, s in enumerate(out):
        s.id = i
    sv = Survey(out, Fs, cf, nfft, M, freq, 10*np.log10(A + 1e-30), float(np.median(10*np.log10(floor[fit.a:fit.b]))))
    sv.floor_db = 10*np.log10(floor)
    return sv

def _net(c, occ, groups, t_start):
    """Steps 4-6 by the learned segmenter (rpps.analysis.spectrum.net), times from t_start"""
    from . import net
    out = net.boxes(c, occ, groups)
    if t_start:
        for s_ in out:
            s_.t0, s_.t1 = s_.t0 + t_start, s_.t1 + t_start
            if s_.bursts:
                s_.bursts = [(u + t_start, v + t_start) for u, v in s_.bursts]
    return out

def _tracked(prev, a, b, lvl, c, t_start, tol_db=0.5):
    """The boxes of the last step's span that this one (a..b, mean level lvl dB) is, still: the same bins (to 2% of its
    width) and level (to tol_db), all steady (none bursting). Copies, their SNR and power measured again over the
    ranges kept; None when it has moved (measured afresh)"""
    tol = max(2, int(0.02*(b - a)))
    for a2, b2, l2, sigs in prev:
        if abs(a - a2) <= tol and abs(b - b2) <= tol and abs(lvl - l2) <= tol_db:
            if any(s.bursts for s in sigs) or not all(hasattr(s, "_range") for s in sigs):
                return None
            import copy
            new = {id(s): copy.copy(s) for s in sigs}
            for s in new.values():
                x0, x1, i0, i1, loc = s._range
                excess = float(np.sum(c.Z[x0:x1]/loc - 1.0))
                if excess > 0:
                    s.snr = float(10*np.log10(excess/(i1 - i0)))
                s.power = float(10*np.log10(c.A[i0:i1].sum()*c.bin_hz + 1e-30))
                s.t0, s.t1 = t_start, t_start + c.T
                if s.over is not None:
                    s.over = new.get(id(s.over), s.over)
            return list(new.values())
    return None

class Finder:
    """Signals in a stream, block by block (a live display): find over the last `frames` frames, the floor refitted every
    `refit` frames [4 x frames: the receiver's passband changes slowly], ids kept from block to block (Survey.match)

        f = Finder(Fs, nfft=4096, cf=cf)
        for blk in source(65536):
            sv = f.step(blk.data)       # a Survey (None before a first frame); times [s] from the stream's start

    A step costs the FFTs of its new frames and a detection over the held ones. Between refits it keeps the detector,
    the mean spectrum (a running sum), the whitened frames and the detections of groups of frames already complete;
    and a span that has not moved since the last step (its bins to 2% of its width, its level to 0.5 dB, none of it
    bursting) keeps its boxes, their SNR and power measured again. Every `full` steps, and after a refit, every span is
    measured afresh (track=False: always). Frames are cut at fixed places in the stream and the floor refitted at fixed
    frame counts, so without tracking what it holds after a stretch of samples doesn't depend on how the stretch came in
    blocks. overlap: of the frames [the effort level's when made]; the rest (pfa, edge, split_db, ...) as find's."""

    def __init__(self, Fs, nfft=4096, cf=0.0, frames=32, refit=None, overlap=None, track=True, full=8, **kw):
        self.Fs, self.nfft, self.cf = float(Fs), int(nfft), cf
        self.frames = int(frames)
        self.refit = max(1, int(4*self.frames if refit is None else refit))
        self.track, self.full = track, max(1, int(full))
        ov = _SETTINGS[_effort.level()]["overlap"] if overlap is None else overlap
        self.hop = max(1, int(self.nfft*(1 - ov)))
        self.kw = dict(pfa=1e-3, edge=0.005, split_db=4.0, drop_db=10.0, min_bins=1, min_across=8, segment="stair")
        self.kw.update({k: v for k, v in kw.items() if k in self.kw})
        self.blocks, self.sys_db = kw.get("blocks", 64), kw.get("sys_db")
        self.reset()

    def reset(self):
        """Forget the stream: the next step starts a new one"""
        self._raw = np.zeros(0, np.complex64) # samples from the oldest held frame's start on
        self._raw0 = 0                        # that start's place in the stream
        self._P = np.zeros((0, self.nfft), np.float32)
        self._Pw = np.zeros((0, self.nfft), np.float32) # the held frames over the floor
        self._sum = np.zeros(self.nfft)       # their sum (the mean spectrum, running)
        self._n = 0                           # frames made so far
        self._fit = None
        self._state = {}
        self._steps = 0
        self.survey = None

    def _refit_now(self):
        self._fit = fit_floor(self._P, self.hop, self.kw["edge"], self.blocks, self.sys_db)
        self._Pw = self._P/self._fit.floor[None, :].astype(np.float32)
        self._sum = self._P.sum(axis=0, dtype=np.float64) # (exact again: no drift from the running sum)
        self._state = {}
        self._steps = 0

    def _add(self, P):
        """Hold these frames (the newest `frames`), refitting the floor at each multiple of `refit` frames made, and
        until enough frames are held"""
        while len(P):
            to = self.refit - self._n % self.refit # frames to the next refit
            take, P = P[:to], P[to:]
            drop = max(0, len(self._P) + len(take) - self.frames)
            self._sum += take.sum(axis=0, dtype=np.float64) - self._P[:drop].sum(axis=0, dtype=np.float64)
            self._P = np.concatenate((self._P, take))[-self.frames:]
            if self._fit is not None:
                self._Pw = np.concatenate((self._Pw, take/self._fit.floor[None, :].astype(np.float32)))[-self.frames:]
            self._n += len(take)
            if self._n % self.refit == 0 or self._fit is None or len(self._P) < self.frames:
                if len(self._P) >= 2:
                    self._refit_now()

    def step(self, block):
        """Take a block of samples; the Survey of the frames now held (None while there are fewer than 2)"""
        x = np.asarray(block, np.complex64)
        self._raw = np.concatenate((self._raw, x))
        start = self._n*self.hop - self._raw0 # where the next frame starts in _raw
        if len(self._raw) - start >= self.nfft:
            P, _ = spectrogram(self._raw[start:], self.Fs, self.nfft, 1 - self.hop/self.nfft)
            self._add(P)
        # Keep the samples from the oldest held frame's start
        first = (self._n - len(self._P))*self.hop
        if first > self._raw0:
            self._raw = self._raw[first - self._raw0:]
            self._raw0 = first
        if len(self._P) < 2 or self._fit is None:
            return None
        M = len(self._P)
        x_held = self._raw[:(M - 1)*self.hop + self.nfft]
        st = self._state
        st["A"], st["Pw"], st["frame0"] = self._sum/M, self._Pw, self._n - M
        if not self.track or self._steps % self.full == 0:
            st.pop("spans", None)
        self._steps += 1
        sv = _find_frames(self._P, self.hop, x_held, self.Fs, self.cf, self._fit, t_start=self._raw0/self.Fs,
                          state=st, **self.kw)
        if self.survey is not None:
            sv.match(self.survey)
        self.survey = sv
        return sv

def _on(c, a, b, local):
    """Step 4: the frames a span is on (None: continuous, or too weak a frame to tell), its bursts [(t0, t1)] and duty.
    The frames say which to average; the bursts' times come from frames r times shorter (r as large as leaves the span
    32 bins wide: about 16 degrees of freedom a frame), gated the same way with one-frame gaps closed, when the span stands
    out of the noise in them too"""
    hi, lo = c.det.thresholds(b - a, 1)
    band = c.Pw[:, a:b].mean(axis=1)/_mean(local, a, b)
    if band.mean() < 2*hi: # a frame alone can't tell it from noise
        return None, None, 1.0
    on = _gate(band, hi, lo)
    if on.all() or not on.any():
        return None, None, 1.0
    T = c.T
    bursts = [(p*c.hop/c.Fs, min(T, ((q - 1)*c.hop + c.nfft)/c.Fs)) for p, q in zip(*_runs(on))]
    duty = float(on.mean())
    r = 1 << int(np.clip(np.floor(np.log2(max((b - a)/32, 1))), 0, np.log2(max(c.nfft//16, 1))))
    if r > 1:
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
            bursts = [(p*nt/c.Fs, min(T, q*nt/c.Fs)) for p, q in zip(*_runs(on_t))]
            duty = float(on_t.mean())
    return on, bursts, duty

def _gate(band, hi, lo):
    """Frames on: runs over `lo` that reach `hi` (hysteresis)"""
    seed, grow = band > hi, band > lo
    cs = np.concatenate(([0], np.cumsum(seed)))
    on = np.zeros(len(band), bool)
    for p, q in zip(*_runs(grow)):
        if cs[q] > cs[p]:
            on[p:q] = True
    return on

def _mean(local, a, b):
    """A local floor's level over bins a..b (local: 1, or a level per bin)"""
    return float(local) if np.isscalar(local) else float(np.mean(local[a:b]))

# --- Steps 5-6: the span's spectrum as a staircase of flat levels (optimal changepoints), its signals read off the levels

def need_of(c):
    """The least notch or step between two signals [dB]: 3/4 of split_db"""
    return 0.75*c.split_db

def _pelt(y, w, beta, lmin=2):
    """Optimal partition of y into flat segments (weights w: 1/variance), each change costing beta: PELT (Killick et al.
    2012), exact, about linear. Returns the segment bounds [0, ..., n]. The C++ kernel when loaded (the loop below is the
    reference it matches)"""
    from ... import _accel
    if _accel.native is not None and hasattr(_accel.native, "pelt"):
        return [int(v) for v in _accel.native.pelt(np.ascontiguousarray(y, np.float64), np.ascontiguousarray(w, np.float64),
                                                   float(beta), int(lmin))]
    return _pelt_py(y, w, beta, lmin)

def _pelt_py(y, w, beta, lmin=2):
    """_pelt's Python loop (the reference)"""
    n = len(y)
    S0 = np.concatenate(([0.0], np.cumsum(w)))
    S1 = np.concatenate(([0.0], np.cumsum(w*y)))
    S2 = np.concatenate(([0.0], np.cumsum(w*y*y)))
    F = np.full(n + 1, np.inf)
    F[0] = -beta
    last = np.zeros(n + 1, int)
    R = np.array([0])
    for t in range(lmin, n + 1):
        ok = t - R >= lmin
        Rv = R[ok]
        if len(Rv) == 0:
            R = np.append(R, t)
            continue
        cost = (S2[t] - S2[Rv]) - (S1[t] - S1[Rv])**2/(S0[t] - S0[Rv])
        vals = F[Rv] + cost
        k = int(np.argmin(vals))
        F[t] = vals[k] + beta
        last[t] = Rv[k]
        R = np.concatenate((R[~ok], Rv[vals <= F[t]], [t]))
    bounds, t = [n], n
    while t > 0:
        t = last[t]
        bounds.append(t)
    return bounds[::-1]

def _span_seg(c, a, b):
    """Steps 4-6 for one occupied span a..b by segmentation:

    - the spectrum (over the frames it is on), with as much again each side for context, in points of d bins (d up to
      16, less for a narrow span: 128 points across it at least), in dB
    - a staircase fitted to it (_pelt): each point weighted by its own scatter (the noise's, or the roughness about it
      when more: a carrier's spectral lines, a rough top), a change costing 3 log n
    - the levels read: runs of levels standing over the floor (significantly, and by up_db) are signals; a run is cut
      at a notch (a level `split_db` x 3/4 under the highest each side) and at a step between two broad levels of
      `split_db` x 3/4 (a carrier flush against a stronger one). A run whose cuts leave broad levels over the floor between
      its pieces is a pedestal: listed with `holds`, its pieces measured over it
    - each piece's edges where the spectrum crosses `drop_db` under its level, or halfway to the level outside it, between
      the levels' bounds; dropped if its edge rises over most of its width (a hump), or it fails the detector at its width
    Only pieces centred in a..b are returned (the context is another span's)."""
    on, bursts, duty = _on(c, a, b, 1.0)
    bp = c.Pw[on if on is not None else slice(None), a:b].mean(axis=1, dtype=np.float64)
    m = max(1, int(bp.sum()**2/np.sum(bp**2)))
    spec = c.Pw[on].mean(axis=0) if on is not None else c.Z
    a_use, b_use = c.cut
    w = b - a
    d = int(np.clip(1 << int(np.floor(np.log2(max(w/128, 1)))), 1, 8)) # (at most 8: notches between carriers are 3-8 bins)
    lo_r = max(a_use, a - max(w, 32*d))
    hi_r = min(b_use, b + max(w, 32*d))
    n = (hi_r - lo_r)//d
    if n < 4:
        return []
    hi_r = lo_r + n*d
    y = 10*np.log10(np.maximum(spec[lo_r:hi_r].reshape(n, d).mean(axis=1), 1e-30))
    # Each point's scatter: the noise's for d bins over m frames, or the roughness about it (robust: half the spread of
    # its neighbours' differences, over 32 points) when more
    var0 = (4.343**2)/_dof(c.nfft, c.hop, m, d)
    dy = np.abs(np.diff(y, prepend=y[0]))
    k = 16
    pad = np.concatenate((np.full(k, dy[0]), dy, np.full(k, dy[-1])))
    med = np.median(np.lib.stride_tricks.sliding_window_view(pad, 2*k + 1), axis=1)[:n]
    var = np.maximum(var0, (1.4826*med)**2/2)
    bnd = _pelt(y, 1/var, 3.0*np.log(max(n, 3)), lmin=2)
    segs = [(s, e, float(np.average(y[s:e], weights=1/var[s:e])), float(np.sqrt(np.mean(var[s:e])/(e - s))))
            for s, e in zip(bnd[:-1], bnd[1:])]
    lev = np.array([g[2] for g in segs])
    up = [(g[2] >= max(c.split_db/4, 5*g[3])) for g in segs] # (over the floor: 0 dB, whitened)
    need = 0.75*c.split_db
    out = []
    i = 0
    while i < len(segs):
        if not up[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(segs) and up[j + 1]:
            j += 1
        out += _seg_run(c, segs, lev, i, j, need, lo_r, d, spec, m, bursts, on, duty, a, b)
        i = j + 1
    if d > 1:
        out += _seg_lines(c, segs, lo_r, d, spec, m, bursts, duty, a, b, out)
    return out

def _own_lines(boxes, least=3, even=0.15):
    """A carrier's own lines go: `least` or more lines standing on it at one spacing (each gap within `even` of their
    median), their middle within one spacing of its centre (what periodic structure in its symbols puts in its spectrum,
    a DVB-S carrier's sync bytes). A tone on a carrier alone, or off its centre, stays (a jammer)"""
    drop = set()
    for s in boxes:
        on = sorted([t for t in boxes if t.over is s and not t.holds and getattr(t, "lines", 0) == 0], key=lambda t: t.mid)
        if len(on) < least:
            continue
        gaps = np.diff([t.mid for t in on])
        g = float(np.median(gaps))
        if g > 0 and np.all(np.abs(gaps - g) <= even*g) and abs(0.5*(on[0].mid + on[-1].mid) - s.mid) <= g:
            drop |= {id(t) for t in on}
            s.holds = max(0, (s.holds or 0) - len(on))
    return [t for t in boxes if id(t) not in drop]

def _combs(c, boxes, least=4, spread=3.0, narrow=12, first=20):
    """Lines side by side at about one spacing (a comb: `least` or more boxes `narrow` bins wide or less, each gap under
    `spread` x the group's median gap, the first under `first` line widths) as one box each, from its first line to its
    last, its SNR its power over the noise across it; the lines, and the slivers of it narrower than half of it, go.
    Across spans: each line of a comb is a span of its own"""
    lines = sorted([t for t in boxes if t.width <= narrow*c.bin_hz and t.over is None], key=lambda t: t.mid)
    if len(lines) < least:
        return boxes
    groups, cur = [], [lines[0]]
    for t in lines[1:]:
        g = t.mid - cur[-1].mid
        gaps = np.diff([u.mid for u in cur]) if len(cur) > 1 else None
        ok = (g <= spread*float(np.median(gaps))) if gaps is not None else (g <= first*max(cur[-1].width, c.bin_hz))
        if ok:
            cur.append(t)
        else:
            groups.append(cur)
            cur = [t]
    groups.append(cur)
    out = list(boxes)
    for grp in groups:
        if len(grp) < least:
            continue
        lo, hi = grp[0].lo, grp[-1].hi
        i0 = int(np.floor(lo/c.bin_hz + c.nfft//2))
        i1 = int(np.ceil(hi/c.bin_hz + c.nfft//2))
        excess = float(np.sum(c.Z[i0:i1] - 1.0))
        if excess <= 0:
            continue
        band = c.A[i0:i1]
        sg = Signal(0.5*(lo + hi), hi - lo, float(np.median(10*np.log10(c.floor[i0:i1]))),
                    float(10*np.log10(max(float(c.Z[i0:i1].max()), 1e-30))), float(10*np.log10(excess/(i1 - i0))), lo=lo,
                    hi=hi, power=float(10*np.log10(band.sum()*c.bin_hz + 1e-30)),
                    peak_freq=float(c.freq[i0 + int(np.argmax(band))]), duty=grp[0].duty, resolved=True, Fs=c.Fs, cf=c.cf,
                    bin_hz=c.bin_hz, clipped=False)
        sg.bursts = None
        sg.over = None
        sg.t0, sg.t1 = min(t.t0 for t in grp), max(t.t1 for t in grp)
        sg.score = float(np.max(c.score[i0:i1]))
        sg._range = (i0, i1, i0, i1, 1.0)
        sg.edge = sg.rough = None
        sg.lines = len(grp) # (how many lines the comb holds)
        ids = {id(t) for t in grp}
        # (its lines, and the line-wide slivers of it the staircase made, go; a carrier among them stays)
        out = [t for t in out if id(t) not in ids and not (lo <= t.mid <= hi and t.width <= narrow*c.bin_hz)]
        out.append(sg)
    return out

def _seg_lines(c, segs, lo_r, d, spec, m, bursts, duty, a, b, boxes):
    """Lines narrower than the staircase's points (tones, a few bins): the detector on the spectrum over the staircase,
    its runs narrower than two points. Each a box of its own, over the box it stands in when it stands in one"""
    db = [g[2] for g in segs]
    # (a level under 3 points is a line's, this pass's: measured over its neighbours' levels, the lower of them)
    under_db = [min(db[k - 1] if k > 0 else db[k + 1], db[k + 1] if k + 1 < len(segs) else db[k - 1])
                if g[1] - g[0] < 3 and len(segs) > 1 else db[k] for k, g in enumerate(segs)]
    under_db = [min(u, x) for u, x in zip(under_db, db)]
    lev = np.repeat([10**(x/10) if x > 0 else 1.0 for x in under_db], [g[1] - g[0] for g in segs])
    under = np.maximum(np.repeat(lev, d), 1.0) # (never under the noise)
    n = len(under)
    resid = spec[lo_r:lo_r + n]/under
    occ, _s, _k = c.det.detect(resid, m)
    out = []
    v = 10*np.log10(np.maximum(spec[lo_r:lo_r + n], 1e-30))
    for s0, e0 in zip(*_runs(occ)):
        if e0 - s0 > 2*d or not (a <= lo_r + (s0 + e0)//2 < b):
            continue
        pk = s0 + int(np.argmax(v[s0:e0]))
        # (over the highest of the staircase beside it, and its edges no further than two points from it: where the
        # staircase lags a steep edge, a run of a few bins is that edge, not a line)
        lo_b, hi_b = max(0, s0 - 2*d), min(n, e0 + 2*d)
        base = 10*np.log10(float(under[lo_b:hi_b].max()))
        top = float(v[pk])
        if top - base < 0.75*c.split_db:
            continue
        thr = base + 0.5*(top - base)
        el = _walk(v, pk, lo_b - 1, thr, -1)
        er = _walk(v, pk, hi_b, thr, +1)
        i0 = int(np.clip(np.floor(lo_r + el + 0.5), 0, c.nfft - 1))
        i1 = int(np.clip(np.ceil(lo_r + er + 0.5), i0 + 1, c.nfft))
        local = 10**(base/10)
        excess = float(np.sum(spec[i0:i1]/local - 1.0))
        # (a line stands 3 dB over what it is on: under that, reviewers found 25 of 28 to be the noise's or a rough
        # top's peaks)
        if excess <= 0 or 10*np.log10(excess/(i1 - i0)) < need_of(c):
            continue
        e_0, e_1 = lo_r + el + 0.5, lo_r + er + 0.5
        lo, hi = (e_0 - c.nfft//2)*c.bin_hz, (e_1 - c.nfft//2)*c.bin_hz
        band = c.A[i0:i1]
        sg = Signal(0.5*(lo + hi), hi - lo, float(np.median(10*np.log10(c.floor[i0:i1]))) + base, top - base,
                    float(10*np.log10(excess/(i1 - i0))), lo=lo, hi=hi, power=float(10*np.log10(band.sum()*c.bin_hz + 1e-30)),
                    peak_freq=float(c.freq[lo_r + pk]), duty=duty, resolved=(e_1 - e_0) >= c.min_across, Fs=c.Fs, cf=c.cf,
                    bin_hz=c.bin_hz, clipped=False)
        sg.bursts = bursts
        sg.over = next((bx for bx in boxes if bx.lo <= sg.mid <= bx.hi), None)
        sg.t0, sg.t1 = (bursts[0][0], bursts[-1][1]) if bursts else (0.0, c.T)
        sg.score = float(np.max(c.score[i0:i1]))
        sg._range = (i0, i1, i0, i1, local)
        sg.edge = sg.rough = None
        out.append(sg)
    return out

def _seg_run(c, segs, lev, i, j, need, lo_r, d, spec, m, bursts, on, duty, a, b, base=0.0):
    """The signals of one run of levels i..j over the floor, or over `base` inside a pedestal (see _span_seg)"""
    run = list(range(i, j + 1))
    L = [segs[k][1] - segs[k][0] for k in run]
    width = sum(L)
    if base == 0.0: # A pedestal: its lowest broad level well over the floor, a third of it at least, nothing dipping
        # far under it (a row of carriers has notches to the floor), something standing on it
        broad = [k for k, l in zip(run, L) if l >= max(2, width//20)]
        if broad:
            lvl0 = min(lev[k] for k in broad)
            cover = sum(l for k, l in zip(run, L) if abs(lev[k] - lvl0) <= 1.5)
            # (between its first and last broad levels: its own sides ramp up from the floor)
            deep = any(lev[k] < lvl0 - need for k in run if broad[0] < k < broad[-1])
            # (what stands on it: groups of levels 3 points wide at least; narrower are lines, the line pass's: a
            # carrier's own spectral lines don't make it a pedestal)
            groups, k = [], 0
            while k < len(run):
                if lev[run[k]] >= lvl0 + 0.5*need:
                    e = k
                    while e + 1 < len(run) and lev[run[e + 1]] >= lvl0 + 0.5*need:
                        e += 1
                    if segs[run[e]][1] - segs[run[k]][0] >= 3 and max(lev[run[q]] for q in range(k, e + 1)) >= lvl0 + need:
                        groups.append((run[k], run[e]))
                    k = e + 1
                else:
                    k += 1
            if lvl0 >= need and cover >= 0.3*width and not deep and groups:
                out = []
                for g0, g1 in groups:
                    out += _seg_run(c, segs, lev, g0, g1, need, lo_r, d, spec, m, bursts, on, duty, a, b, base=float(lvl0))
                if out:
                    ped = _seg_piece(c, segs, lev, run, run, lo_r, d, spec, m, bursts, on, duty, 0.0, holds=True)
                    if ped is not None and a <= (ped.mid/c.bin_hz + c.nfft//2) < b:
                        ped.holds = len(out)
                        for t in out:
                            if t.over is None:
                                t.over = ped
                        out.append(ped)
                    return out
    # Notches: a level `need` under the highest broad level each side within the run (levels 3 points wide at least:
    # a carrier between its own spectral lines is not in a notch)
    wide = np.array([lev[k] if l >= 3 else -np.inf for k, l in zip(run, L)])
    cut = set()
    for p, k in enumerate(run[1:-1], 1):
        if min(wide[:p].max(), wide[p + 1:].max()) - lev[k] >= need:
            # (a run of low levels is one notch: cut only at its lowest)
            if lev[k] <= lev[run[p - 1]] and lev[k] <= lev[run[p + 1]]:
                cut.add(k)
    # Steps: two broad levels side by side `need` apart (a carrier flush against a stronger one)
    steps = set()
    broad = max(4, width//8)
    for p in range(len(run) - 1):
        k0, k1 = run[p], run[p + 1]
        if k0 in cut or k1 in cut:
            continue
        lo_len, hi_len = (L[p], L[p + 1]) if lev[k0] < lev[k1] else (L[p + 1], L[p])
        # (both broad, the lower a quarter of the higher at least: a terrace, not a level of a skirt falling away)
        if abs(lev[k0] - lev[k1]) >= need and min(L[p], L[p + 1]) >= broad and lo_len >= 0.25*hi_len:
            steps.add(p + 1) # (a boundary before run[p+1])
    # Pieces: runs of levels between notches, split again at steps
    pieces, cur = [], []
    for p, k in enumerate(run):
        if k in cut:
            if cur:
                pieces.append(cur)
            cur = []
            continue
        if p in steps and cur:
            pieces.append(cur)
            cur = []
        cur.append(k)
    if cur:
        pieces.append(cur)
    out = []
    for pc in pieces:
        # Bumps: levels standing `need` over the levels either side of them inside the piece (a narrower signal on a
        # wider one: a jammer on a carrier), each a box of its own over the piece, the piece measured without them
        # (the piece's own level T: its longest level. A run of levels `need` over T, with levels about T on both sides
        # of it inside the piece, narrower than half of it: a bump. A carrier's own top over its skirts has no level of
        # T beyond it)
        pw = segs[pc[-1]][1] - segs[pc[0]][0]
        T = lev[max(pc, key=lambda q: segs[q][1] - segs[q][0])]
        at_T = [abs(lev[q] - T) < c.split_db and lev[q] < T + need for q in pc]
        bumps, q = [], 0
        while q < len(pc):
            if lev[pc[q]] >= T + need:
                e = q
                while e + 1 < len(pc) and lev[pc[e + 1]] >= T + need:
                    e += 1
                # (3 points wide at least: narrower is a line, the line pass's, which never makes a carrier `holds`: a
                # DVB-S carrier's lines from its periodic sync bytes would make it a "pedestal" profile skips)
                if any(at_T[:q]) and any(at_T[e + 1:]) and 3 <= segs[pc[e]][1] - segs[pc[q]][0] < 0.5*pw:
                    bumps.append(pc[q:e + 1])
                q = e + 1
            else:
                q += 1
        under = [q for q in pc if not any(q in bm for bm in bumps)]
        s = _seg_piece(c, segs, lev, pc, run, lo_r, d, spec, m, bursts, on, duty, base,
                       top_from=under if bumps else None, top_level=float(T) if bumps else None)
        if s is not None and a <= (s.mid/c.bin_hz + c.nfft//2) < b:
            out.append(s)
            for bm in bumps:
                # (over T: its edges toward it, its SNR over it)
                t = _seg_piece(c, segs, lev, bm, pc, lo_r, d, spec, m, bursts, on, duty, float(T), over_level=float(T))
                if t is not None:
                    t.over = s
                    s.holds = getattr(s, "holds", 0) + 1
                    out.append(t)
    return out

def _seg_piece(c, segs, lev, pc, run, lo_r, d, spec, m, bursts, on, duty, base, holds=False, top_from=None,
               over_level=None, top_level=None):
    """One piece (levels pc of the run) measured: its level, what is outside it each side, its edges on the spectrum,
    the checks, the Signal"""
    s0, s1 = segs[pc[0]][0], segs[pc[-1]][1]
    ls = [segs[k][1] - segs[k][0] for k in pc]
    if holds: # a pedestal's level: its base, the levels between its pieces (the lowest broad ones)
        top = float(np.min([lev[k] for k, l in zip(pc, ls) if l >= max(ls)/4]))
        k_top = pc[len(pc)//2]
    else: # its level: the broadest of its levels within half a split of its highest (not counting what stands on it,
        # nor levels under 3 points: a line on it is the line pass's)
        src = top_from or pc
        ls_src = [segs[k][1] - segs[k][0] for k in src]
        hi = max((lev[k] for k, l in zip(src, ls_src) if l >= 3), default=max(lev[k] for k in src))
        _, top, k_top = max((l, lev[k], k) for k, l in zip(src, ls_src) if lev[k] >= hi - 0.5*need_of(c))
        top = float(top)
        if top_level is not None: # (something stands on it: its level is its longest one)
            k_top = max(src, key=lambda q: segs[q][1] - segs[q][0])
            top = top_level
    i_first, i_last = run.index(pc[0]), run.index(pc[-1])
    out_l = float(lev[run[i_first - 1]]) if i_first > 0 else max(base, 0.0)
    out_r = float(lev[run[i_last + 1]]) if i_last + 1 < len(run) else max(base, 0.0)
    if i_first == 0 and pc[0] > 0:
        out_l = max(base, min(float(lev[pc[0] - 1]), out_l)) if not holds else 0.0
    if i_last + 1 == len(run) and pc[-1] + 1 < len(segs):
        out_r = max(base, min(float(lev[pc[-1] + 1]), out_r)) if not holds else 0.0
    if holds:
        out_l = out_r = 0.0
    if over_level is not None: # (standing on a wider signal: what is outside it is that signal's level)
        out_l = out_r = over_level
    abut_l, abut_r = out_l > top - 1.0, out_r > top - 1.0 # (flush against something higher: ends at the step)
    thr_l = max(top - c.drop_db, out_l + 0.5*(top - out_l))
    thr_r = max(top - c.drop_db, out_r + 0.5*(top - out_r))
    # On the spectrum smoothed by d (the points' bins), within a level of the piece's bounds each side
    v = 10*np.log10(np.maximum(_box(spec[lo_r:lo_r + segs[-1][1]*d], d), 1e-30))
    lo_lim = segs[pc[0] - 1][0]*d if pc[0] > 0 else 0
    hi_lim = segs[pc[-1] + 1][1]*d if pc[-1] + 1 < len(segs) else len(v)
    p0, p1 = s0*d, s1*d
    core = p0 + int(np.argmax(_box(v[p0:p1], max(1, (p1 - p0)//8)))) if p1 > p0 else p0
    g = max(2, 2*d)
    el = float(p0) if abut_l else _walk(v, core, lo_lim - 1, thr_l, -1, g)
    er = float(p1 - 1) if abut_r else _walk(v, core, hi_lim, thr_r, +1, g)
    el, er = max(el, lo_lim), min(er, hi_lim - 1)
    if er - el < c.min_bins:
        return _why(c, 'narrow', lo_r + el, lo_r + er)
    a_use, b_use = c.cut
    # A hump: its level rises over most of its width (on the staircase: from 1/4 to 3/4 of its height over what is
    # outside, wider than a fifth of it or 2% of the band). Sides at a notch or step are a neighbour's; not checked
    if not holds:
        stair = np.repeat(lev, [g_[1] - g_[0] for g_ in segs])
        span_ok = max(0.2*(er - el), 0.02*c.nfft)/d
        sm = 10*np.log10(np.maximum(_box(spec[lo_r:lo_r + len(stair)*d], max(d, int((er - el)/32))), 1e-30))
        for side, outv, edge_pt, step in (("left", out_l, int(el)//d, -1), ("right", out_r, int(er)//d, +1)):
            facing_floor = ((i_first == 0) if step < 0 else (i_last + 1 == len(run))) or over_level is not None
            if not facing_floor or (step < 0 and lo_r + el <= a_use + 1) or (step > 0 and lo_r + er >= b_use - 2):
                continue
            hl, ll = 10**(top/10), 10**(outv/10)
            t25, t75 = 10*np.log10(ll + 0.25*(hl - ll)), 10*np.log10(ll + 0.75*(hl - ll))
            mid_pt = (segs[k_top][0] + segs[k_top][1])//2 # (its own level's middle, not what stands on it)
            # From just outside it inward (a top that wanders under 3/4 of its level would stop a walk from the middle):
            # where it first reaches 1/4 and 3/4 of its height. Outside, the staircase must be under 3/4 of it
            start = segs[pc[0]][0] - 1 if step < 0 else segs[pc[-1]][1]
            outside_ok = 0 <= start < len(stair)
            below75 = [start] if outside_ok and stair[start] < t75 else []
            if not len(below75):
                at_band_edge = (lo_r <= a_use + d) if step < 0 else (lo_r + len(stair)*d >= b_use - d)
                if at_band_edge: # (cut by the band's edge: Signal.clipped. With one edge to judge it by, it stands over
                    # what is outside the other by a split at least)
                    other = out_r if step < 0 else out_l
                    if top - other < need_of(c):
                        return _why(c, f'weak at the band edge ({side})', lo_r + el, lo_r + er)
                    continue
                return _why(c, f'no edge ({side})', lo_r + el, lo_r + er) # (it never falls: part of something wider)
            # How fast it rises, on the spectrum itself smoothed over 1/32 of its width (the staircase has a few levels
            # across a gentle hump, its first step from the floor as abrupt as a carrier's edge)
            b0, b1 = start*d + (d - 1 if step < 0 else 0), mid_pt*d
            idx = np.arange(max(b0, 0), b1 + 1) if step < 0 else np.arange(min(b0, len(sm) - 1), b1 - 1, -1)
            r25, r75 = idx[sm[idx] >= t25], idx[sm[idx] >= t75]
            if len(r25) and len(r75) and abs(int(r75[0]) - int(r25[0])) > span_ok*d:
                return _why(c, f'rounded edge ({side})', lo_r + el, lo_r + er)
    local = 10**(base/10) if base > 0 else 1.0
    i0 = int(np.clip(np.floor(lo_r + el + 0.5), 0, c.nfft - 1))
    i1 = int(np.clip(np.ceil(lo_r + er + 0.5), i0 + 1, c.nfft))
    z = spec[i0:i1]/local
    if not c.det.passes(z, m):
        return _why(c, 'under the threshold at its width', lo_r + el, lo_r + er)
    # SNR: its power over what is under it, summed out to where it is within a tenth of its height of what is outside
    xl = el if abut_l else _walk(v, int(el), lo_lim - 1, out_l + 0.1*(top - out_l), -1)
    xr = er if abut_r else _walk(v, int(er), hi_lim, out_r + 0.1*(top - out_r), +1)
    half = 0.5*(er - el)
    x0 = int(np.clip(np.floor(lo_r + max(xl, el - half) + 0.5), 0, i0))
    x1 = int(np.clip(np.ceil(lo_r + min(xr, er + half) + 0.5), i1, c.nfft))
    excess = float(np.sum(spec[x0:x1]/local - 1.0))
    if excess <= 0:
        return None
    n = i1 - i0
    snr = 10*np.log10(excess/n)
    e0, e1 = lo_r + el + 0.5, lo_r + er + 0.5
    lo, hi = (e0 - c.nfft//2)*c.bin_hz, (e1 - c.nfft//2)*c.bin_hz
    band = c.A[i0:i1]
    sg = Signal(0.5*(lo + hi), hi - lo, float(np.median(10*np.log10(c.floor[i0:i1]))) + base,
                float(10*np.log10(max(z.max(), 1e-30))), float(snr), lo=lo, hi=hi,
                power=float(10*np.log10(band.sum()*c.bin_hz + 1e-30)), peak_freq=float(c.freq[i0 + int(np.argmax(band))]),
                duty=duty, resolved=(e1 - e0) >= c.min_across, Fs=c.Fs, cf=c.cf, bin_hz=c.bin_hz,
                clipped=e0 <= a_use + d + 1 or e1 >= b_use - d - 1) # (to a point of the usable band's ends)
    sg.bursts = bursts
    sg.over = None
    sg.t0, sg.t1 = (bursts[0][0], bursts[-1][1]) if bursts else (0.0, c.T)
    sg.score = float(np.max(c.score[i0:i1]))
    sg._range = (x0, x1, i0, i1, local)
    sg.edge = None
    sg.rough = None
    return sg
