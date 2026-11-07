from types import SimpleNamespace

import numpy as np

from ...base import Meta, Stage
from ...filters.shaping import rrc_pulse

def matrix_mf(x, centres, sps, beta=0.2, span=8):
    """Matched filter + symbol sampling as a (banded) matrix: y = W @ x

    W[i, j] = rrc((j - centres[i]) / sps), so symbols are taken at any fractional instant
    and sps may be fractional. Only the 2*span*sps columns that matter are built per row.
    x[centres +/- span*sps] must exist. Weights have unit energy per row.
    """
    S = span * sps
    j0 = np.ceil(centres - S).astype(np.int64)
    idx = j0[:, None] + np.arange(int(np.ceil(2*S)) + 2)
    w = rrc_pulse((idx - centres[:, None]) / sps, beta)
    w /= np.sqrt(np.sum(w**2, axis=1, keepdims=True))
    return np.sum(w * x[idx], axis=1)

def centres_in(n, t0, sps, span=8):
    """Symbol centres t0 + k*sps whose whole support fits in n samples"""
    last = n - span*sps - 5
    if last < t0:
        return np.zeros(0)
    return t0 + sps*np.arange(int((last - t0)//sps) + 1)

def best_tau(x, sps, beta=0.2, span=8, lo=0, hi=None, grid=16):
    """Symbol timing in [0, sps) maximising the matched-filter output energy

    The symbol lattice is tau + k*sps in absolute sample index, so results from different
    segments of x are comparable. Returns (tau, mean |y|^2)
    """
    hi = len(x) if hi is None else hi
    seg = x[:hi]
    S = span*sps
    energy = np.zeros(grid)
    for g in range(grid):
        tau = g*sps/grid
        k0 = int(np.ceil((lo + S + 5 - tau)/sps))
        c = centres_in(hi, tau + k0*sps, sps, span)
        energy[g] = np.mean(np.abs(matrix_mf(seg, c, sps, beta, span))**2) if len(c) else 0.0
    g = int(np.argmax(energy))
    a, b, c_ = energy[(g-1) % grid], energy[g], energy[(g+1) % grid]
    den = a - 2*b + c_
    d = 0.0 if den == 0 else 0.5*(a - c_)/den
    return ((g + d)*sps/grid) % sps, float(b)

class Mat(Stage):
    """Matrix matched filter and symbol timing

    Needs meta.stream.det (see rpps.analysis.modulation) for beta and the timing offset. Symbols are
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
        det = meta.stream.det
        S = self.span*meta.sps
        st.t_next = det.tau + np.ceil((S + 5 - det.tau)/meta.sps)*meta.sps
        st.beta = self.beta or det.beta
        st.det = det

    def step(self, samples, st, meta: Meta):
        if meta.stream.det is None: # Still acquiring
            return samples[:0]
        if st.det is None:
            self._start(st, meta)
        sps = meta.sps
        if st.det is not meta.stream.det:
            # Refined estimates: move to their lattice (absolute index = st.base + buffer index)
            st.det = meta.stream.det
            k = np.ceil((st.base + st.t_next - st.det.tau)/sps)
            st.t_next = st.det.tau + k*sps - st.base
        samples = samples if st.buffer is None else np.concatenate((st.buffer, samples))
        centres = centres_in(len(samples), st.t_next, sps, self.span)

        out = np.zeros(0, dtype=np.complex64)
        nxt = st.t_next
        if len(centres):
            out = matrix_mf(samples, centres, sps, st.beta, self.span)
            nxt = centres[-1] + sps
            if self.track and len(centres) >= 64:
                # Energy at +/- DELTA around the lattice: the parabola vertex is the timing error
                e = [np.mean(np.abs(matrix_mf(samples, centres + d, sps, st.beta, self.span))**2)
                     for d in (-self.DELTA, self.DELTA)]
                e0 = np.mean(np.abs(out)**2)
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
