from types import SimpleNamespace

import numpy as np

from ... import _accel
from ...base import Meta, Stage
from ...signal.filters.shaping import rrc_pulse
from ...math.interp import peak, fractional, REACH

def matrix_mf(x, centres, sps, beta=0.2, span=8):
    """Matched filter + symbol sampling as a (banded) matrix: y = W @ x

    W[i, j] = rrc((j - centres[i]) / sps), so symbols are taken at any fractional instant
    and sps may be fractional. Only the 2*span*sps columns that matter are built per row.
    x[centres +/- span*sps] must exist (`fits`; IndexError otherwise). Weights have unit energy per row.
    Complex x runs in C++ when the kernels are loaded (rpps._native.matrix_mf: the rows computed and applied in one
    pass, never stored); the code below is the reference it agrees with to rounding.
    """
    native = _accel.native
    if native is not None and isinstance(x, np.ndarray) and x.dtype in (np.complex64, np.complex128):
        return native.matrix_mf(np.ascontiguousarray(x), np.ascontiguousarray(centres, dtype=np.float64), float(sps),
                                float(beta), float(span))
    S = span * sps
    j0 = np.ceil(centres - S).astype(np.int64)
    lags = np.arange(int(np.ceil(2*S)) + 2)
    if len(j0) and (j0.min() < 0 or j0.max() + len(lags) > len(x)): # (numpy would wrap a negative index round)
        raise IndexError("matrix_mf: a symbol's support runs outside x")
    idx = j0[:, None] + lags
    w = _rrc_rows(j0 - np.asarray(centres, dtype=np.float64), lags, sps, beta)
    w /= np.sqrt(np.sum(w**2, axis=1, keepdims=True))
    return np.sum(w * x[idx], axis=1)

def fits(centres, n, sps, span=8):
    """Which centres have their whole matched-filter support in n samples (what matrix_mf reads)"""
    S = span*sps
    j0 = np.ceil(np.asarray(centres, dtype=np.float64) - S)
    return (j0 >= 0) & (j0 + int(np.ceil(2*S)) + 2 <= n)

def _rrc_rows(off, lags, sps, beta):
    """rrc_pulse((off[:, None] + lags) / sps, beta), with the sines and cosines of the rows (N) and of the lags (L)
    combined by angle addition instead of N L of each: the same values to rounding. The two limits of the formula
    (t = 0, |t| = 1/(4 beta)) are left to rrc_pulse where a point falls on them"""
    t = (off[:, None] + lags)/sps
    if beta == 0:
        return rrc_pulse(t, beta)
    a, b = np.pi*(1 - beta)/sps, np.pi*(1 + beta)/sps
    sa = (np.sin(a*off)[:, None]*np.cos(a*lags) + np.cos(a*off)[:, None]*np.sin(a*lags)) # sin(pi (1 - beta) t)
    cb = (np.cos(b*off)[:, None]*np.cos(b*lags) - np.sin(b*off)[:, None]*np.sin(b*lags)) # cos(pi (1 + beta) t)
    edge = (np.abs(t) < 1e-9) | (np.abs(np.abs(t) - 1/(4*beta)) < 1e-9)
    with np.errstate(divide="ignore", invalid="ignore"):
        w = (sa + 4*beta*t*cb)/(np.pi*t*(1 - (4*beta*t)**2))
    if edge.any():
        w[edge] = rrc_pulse(t[edge], beta)
    return w

def centres_in(n, t0, sps, span=8):
    """Symbol centres t0 + k*sps whose whole support fits in n samples"""
    last = n - span*sps - 5
    if last < t0:
        return np.zeros(0)
    return t0 + sps*np.arange(int((last - t0)//sps) + 1)

def best_tau(x, sps, beta=0.2, span=8, lo=0, hi=None, grid=16):
    """Symbol timing in [0, sps) maximising the matched-filter output energy, from x[lo:hi]

    The symbol lattice is tau + k*sps in absolute sample index, so results from different
    segments of x are comparable. Returns (tau, mean |y|^2)

    The matched filter runs once, as a convolution; its output is band-limited, so each of the `grid` candidate timings
    reads it at its own symbol instants with a 12-tap windowed sinc (math.interp.fractional: -56 dB at 2 samples a
    symbol), in chunks. A segment too short for the convolution's edges takes the filter matrix for each candidate
    (`_best_tau_rows`), which gives the same answer.
    """
    hi = len(x) if hi is None else hi
    L = int(np.ceil(span*sps))
    if hi - lo < 4*L + 8*sps:
        return _best_tau_rows(x, sps, beta, span, lo, hi, grid)
    h = rrc_pulse(np.arange(-L, L + 1)/sps, beta)
    h /= np.sqrt(np.sum(h**2))
    z = np.convolve(np.asarray(x[lo:hi], dtype=np.complex128), h, mode="valid") # z[i]: output at sample lo + L + i
    z = z.astype(np.complex64) # (the interpolator's native path; its error is far above single precision's)
    taps = 2*REACH
    t0, t1 = lo + L + REACH, lo + L + len(z) - REACH - 1 # Absolute positions the interpolator reads whole
    taus = np.arange(grid)*sps/grid
    ks = [np.arange(np.ceil((t0 - tau)/sps), np.ceil((t1 - tau)/sps)) for tau in taus]
    pos = np.concatenate([tau + k*sps for tau, k in zip(taus, ks)]) - (lo + L)
    count = np.array([len(k) for k in ks], dtype=np.float64)
    which = np.repeat(np.arange(grid), [len(k) for k in ks])
    energy = np.zeros(grid)
    for a in range(0, len(pos), TAU_CHUNK):
        y = fractional(z, pos[a:a + TAU_CHUNK], taps)
        energy += np.bincount(which[a:a + TAU_CHUNK], weights=(y.real.astype(np.float64)**2 + y.imag.astype(np.float64)**2),
                              minlength=grid)
    energy = np.divide(energy, count, out=np.zeros(grid), where=count > 0)
    g, d = peak(energy, circular=True, log=False)
    return ((g + d)*sps/grid) % sps, float(energy[g])

TAU_CHUNK = 1 << 16 # Symbol instants interpolated at a time by best_tau

def _best_tau_rows(x, sps, beta=0.2, span=8, lo=0, hi=None, grid=16):
    """best_tau by the filter matrix, rebuilt for each candidate timing: for segments too short to convolve"""
    hi = len(x) if hi is None else hi
    seg = x[:hi]
    S = span*sps
    energy = np.zeros(grid)
    for g in range(grid):
        tau = g*sps/grid
        k0 = int(np.ceil((lo + S + 5 - tau)/sps))
        c = centres_in(hi, tau + k0*sps, sps, span)
        energy[g] = np.mean(np.abs(matrix_mf(seg, c, sps, beta, span))**2) if len(c) else 0.0
    g, d = peak(energy, circular=True, log=False)
    return ((g + d)*sps/grid) % sps, float(energy[g])

class Mat(Stage):
    """Matrix matched filter and symbol timing

    Needs meta.det (see rpps.sync.detect) for beta and the timing offset. Symbols are
    taken from a lattice t_next + k*sps in the (fractional) sps of Meta, then the lattice is nudged by a
    block-wise energy-maximum estimate (`track` = fraction of it applied) to follow the clock.

    beta/span/track: settings; beta None takes the detected one.
    """
    __slots__ = ("beta", "span", "track")
    DELTA = 0.25 # Samples, offset used to measure timing error

    def __init__(self, beta=None, span=8, track=0.1):
        self.beta = beta
        self.span = span
        self.track = track

    def new_state(self, meta: Meta):
        return SimpleNamespace(buffer=None, t_next=0.0, beta=self.beta, base=0, det=None) # det None: not started

    def _start(self, st, meta: Meta):
        det = meta.det
        S = self.span*meta.sps
        st.t_next = det.tau + np.ceil((S + 5 - det.tau)/meta.sps)*meta.sps
        st.beta = self.beta or det.beta
        st.det = det

    def step(self, samples, st, meta: Meta):
        if meta.det is None: # Still acquiring
            return samples[:0]
        if st.det is None:
            self._start(st, meta)
        sps = meta.sps
        if st.det is not meta.det:
            # Refined estimates: move to their lattice (absolute index = st.base + buffer index), its point nearest the
            # next symbol. (Not the first point at or after it: a refined timing a hair earlier would skip a symbol)
            st.det = meta.det
            k = np.round((st.base + st.t_next - st.det.tau)/sps)
            st.t_next = st.det.tau + k*sps - st.base
        samples = samples if st.buffer is None else np.concatenate((st.buffer, samples))
        centres = centres_in(len(samples), st.t_next, sps, self.span)

        out = np.zeros(0, dtype=np.complex64)
        nxt = st.t_next
        if len(centres):
            out = matrix_mf(samples, centres, sps, st.beta, self.span)
            nxt = centres[-1] + sps
            if self.track and len(centres) >= 64:
                # Energy at +/- DELTA around the lattice: the parabola vertex is the timing error. All three over the
                # symbols whose shifted support is still in the buffer (the first one's can start a sample before it)
                ok = fits(centres - self.DELTA, len(samples), sps, self.span) & fits(centres + self.DELTA, len(samples), sps, self.span)
                e = [np.mean(np.abs(matrix_mf(samples, centres[ok] + d, sps, st.beta, self.span))**2)
                     for d in (-self.DELTA, self.DELTA)]
                e0 = np.mean(np.abs(out[ok])**2)
                den = e[0] - 2*e0 + e[1]
                if den < 0: # A maximum
                    nxt += self.track*np.clip(self.DELTA*(e[0] - e[1])/(2*den), -self.DELTA, self.DELTA)

        # Keep what the next symbol's support needs
        keep = min(max(0, int(np.floor(nxt - self.span*sps)) - 1), len(samples))
        st.buffer = samples[keep:]
        st.t_next = nxt - keep
        st.base += keep

        meta.op.Fs = meta.Fs / sps
        meta.op.sps = 1
        return out.astype(np.complex64, copy=False)
