"""Spectral inversion (a stream conjugated) found by known symbols turning backwards, and undone (rpps.sync)

A high-side LO or I and Q swapped anywhere on the way conjugates a stream.

    symbols + rp.sync.Inversion(rp.symbol.load_header("dvbs2_pl").preamble)      # {"sync": "inversion", "header": ...}

Conjugated, everything turns clockwise. That is not a bit map: random data on a constellation the same conjugated (every
PSK, QAM, APSK) looks the same either way, so only a known sequence that turns (steps not all 0 / 180 degrees) or a code
on a standard's map (`CodeDetection.inverted`) tells. `sense(z, known)`: the symbol steps z[n] conj(z[n-1]) against
the sequence's, forwards and conjugated; differential, so before carrier sync.
"""
from types import SimpleNamespace

import numpy as np

from ..base import Stage
from ..base.control import publish

def _steps(known):
    """The phase steps of a known sequence, k[n] conj(k[n-1]), unit size"""
    k = np.asarray(known, dtype=np.complex128).reshape(-1)
    s = k[1:]*np.conj(k[:-1])
    return s/np.maximum(np.abs(s), 1e-30)

def handed(known):
    """Whether a sequence turns (a step not 0 or 180 degrees): only then does it tell the senses apart"""
    return bool(np.any(np.abs(_steps(known).imag) > 1e-6))

def _matches(z, ref, threshold):
    """(forward, backward, size) correlations of z's steps with ref's, at every start, and the peaks: positions over
    `threshold` that are the largest within the sequence's length"""
    z = np.asarray(z, dtype=np.complex128)
    if len(z) < len(ref) + 1:
        return None
    d = z[1:]*np.conj(z[:-1])
    d = d/(np.abs(d) + 1e-30)
    up = np.correlate(d, ref, "valid")/len(ref) # (np.correlate conjugates its second argument)
    down = np.correlate(d, np.conj(ref), "valid")/len(ref)
    size = np.maximum(np.abs(up), np.abs(down)) # (the same for steps of +-90 degrees only, where conj(ref) = -ref)
    L = len(ref)
    peaks = [j for j in np.nonzero(size > threshold)[0] if size[j] >= size[max(j - L, 0):j + L + 1].max()]
    return up, down, size, np.asarray(peaks, dtype=np.int64)

def sense(z, known, threshold=0.6, top=8):
    """Whether symbols z are inverted, by a known sequence in them: True, False, None (not found, or it doesn't turn)

    The `top` strongest matches over `threshold` vote: True when they turn backwards"""
    if not handed(known):
        return None
    m = _matches(z, _steps(known), threshold)
    if m is None or not len(m[3]):
        return None
    up, down, size, peaks = m
    best = peaks[np.argsort(-size[peaks])[:top]]
    return bool(np.sum(down[best].real) > np.sum(up[best].real))

class Inversion(Stage):
    """Spectral inversion of timed symbols by a known sequence that turns; conjugates them back

    Windows of `window` symbols at absolute positions: each match over `threshold` (forwards or backwards) adds
    Re(forward) - Re(backward) to a score held within +-2 `evidence` (about +-2 a pi/2-BPSK word). The sense is decided
    at +-`evidence` and changes only at the other: a stray match never flips it. Windows wait up to `hold` symbols for the
    first decision, then go out as received. Each goes out in the sense known when it was searched (held ones in the
    first decision's): the output is the input one for one, however cut. Publishes {"inverted"} where the sense is decided
    or changes; est[name]: inverted, score, matches, windows, latency. Real levels pass (their polarity is a frame sync's).
    """
    def __init__(self, known, window=4096, threshold=0.7, evidence=3.0, hold=2**16, name="inversion"):
        if not handed(known):
            raise ValueError("Inversion: the known sequence never turns (steps of 0 or 180 degrees only): it looks the same conjugated")
        self.known = np.asarray(known, dtype=np.complex128).reshape(-1)
        self.ref = _steps(self.known)
        self.window, self.threshold, self.evidence = int(window), float(threshold), float(evidence)
        self.hold, self.name = int(hold), name
        if self.window < 1:
            raise ValueError("Inversion: window >= 1")

    def __str__(self):
        return f"Inversion({self.name})"

    def pos_rate(self, decode):
        return 1.0

    def new_state(self, meta):
        return SimpleNamespace(buf=np.zeros(0, np.complex128), base=None, origin=None, k=0, out=None, pending=[],
                               score=0.0, sense=None, matches=0, real=None)

    def step(self, x, st, meta):
        x = np.asarray(x)
        if st.real is None:
            st.real = x.dtype.kind != "c"
        if st.real:
            meta.stream.est[self.name] = dict(inverted=None, score=0.0, matches=0, windows=0, latency=0.0)
            return x
        if st.base is None:
            st.base = st.origin = st.out = meta.pos
        st.buf = np.concatenate((st.buf, x.astype(np.complex128)))
        end, W, L = st.base + len(st.buf), self.window, len(self.known)
        while True: # Every window whose matches are all in: starts in [a, a + W), each needing L symbols from its start
            a = st.origin + st.k*W
            if end < a + W + L - 1 and not (meta.flush and end > a):
                break
            seg = st.buf[a - st.base:min(a + W + L - 1, end) - st.base]
            self._search(seg, a, st, meta)
            st.pending.append((a, min(a + W, end), st.sense))
            st.k += 1
        out = self._release(st, meta.flush)
        keep = min(st.out, st.origin + st.k*W) - st.base # (what is still to go out, and the next window's start)
        st.buf, st.base = st.buf[keep:], st.base + keep
        meta.stream.est[self.name] = dict(inverted=st.sense, score=st.score, matches=st.matches, windows=st.k,
                                          latency=(end - st.out)/meta.Fs if meta.Fs else 0.0)
        return (np.concatenate(out) if out else np.zeros(0, np.complex128)).astype(x.dtype if x.dtype.kind == "c" else np.complex64)

    def _search(self, seg, a, st, meta):
        m = _matches(seg, self.ref, self.threshold)
        if m is None:
            return
        up, down, _, peaks = m
        for j in peaks:
            st.matches += 1
            st.score = float(np.clip(st.score + up[j].real - down[j].real, -2*self.evidence, 2*self.evidence))
            was = st.sense
            if st.score >= self.evidence:
                st.sense = False
            elif st.score <= -self.evidence:
                st.sense = True
            if st.sense is not None and st.sense != was:
                publish(meta, self, {"inverted": st.sense}, at=a + int(j))

    def _release(self, st, flush):
        """The windows that can go out: all once the sense is known (the undecided ones in it), else those past `hold`"""
        out = []
        while st.pending:
            a, b, s = st.pending[0]
            if s is None and st.sense is None and not flush and st.pending[-1][1] - a <= self.hold:
                break
            st.pending.pop(0)
            s = st.sense if s is None else s
            seg = st.buf[a - st.base:b - st.base]
            out.append(np.conj(seg) if s else seg)
            st.out = b
        return out
