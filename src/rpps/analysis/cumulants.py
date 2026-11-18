"""Higher-order cumulants of symbols: which constellation, blind to Gaussian noise and to the carrier phase

    r = rp.analysis.cumulants.classify(symbols)      # symbol-spaced samples (timing recovered; carrier need not be)
    print(r[0])        # e.g. "16QAM: distance 0.02, Es/N0 14.8 dB"   (best first)
    rp.analysis.cumulants.measure(symbols)           # {"C40": .., "C42": .., "C63": .., "C80": ..} normalized

C_pq is the cumulant of order p with q of its factors conjugated: the moment M_pq = E[x^(p-q) conj(x)^q] less what the
lower orders explain (the moment-cumulant formula over the set partitions of the p factors). Gaussian noise has none above
order 2, so C40, C42, C63, C80 of signal plus noise are the signal's; only C21 (the power) has the noise in it. A carrier
phase turns C_pq by (p - 2q) times it: C42 and C63 (as many conjugated as not) do not turn, the others are compared by
magnitude. Normalized by the signal power to the p/2, each constellation has its own values (Swami & Sadler 2000): BPSK
C42 -2, QPSK -1, 8PSK -1 with C40 0, 16QAM -0.68, 64QAM -0.62, OFDM and noise 0.

`theory` computes them for any point set, so every constellation rpps knows is a candidate. `classify` fits, for each, the
signal's share of the measured power (what the noise does not have) and compares; the share is an Es/N0.
"""
import itertools
from functools import lru_cache

import numpy as np

FEATURES = (("C40", 4, 0), ("C42", 4, 2), ("C63", 6, 3), ("C80", 8, 0), ("C84", 8, 4))

@lru_cache(maxsize=None)
def _partitions(p):
    """Set partitions of range(p), as tuples of blocks"""
    def rec(items):
        if not items:
            yield ()
            return
        first, rest = items[0], items[1:]
        for part in rec(rest):
            for i in range(len(part)):
                yield part[:i] + ((first,) + part[i],) + part[i + 1:]
            yield ((first,),) + part
    return tuple(rec(tuple(range(p))))

def _moment_table(x, p):
    """E[x^a conj(x)^b] for a + b <= p"""
    x = np.asarray(x, dtype=np.complex128)
    xc = np.conj(x)
    M = {}
    pw = [np.ones_like(x)]
    for _ in range(p):
        pw.append(pw[-1]*x)
    for a in range(p + 1):
        for b in range(p + 1 - a):
            M[a, b] = np.mean(pw[a]*pw[b].conj()) if b else np.mean(pw[a])
    return M

def cumulant(M, p, q):
    """C_pq from a moment table: sum over partitions of (-1)^(k-1) (k-1)! prod of the blocks' moments"""
    conj = [i >= p - q for i in range(p)] # the last q factors are conjugated
    total = 0j
    for part in _partitions(p):
        k = len(part)
        term = (-1)**(k - 1)*float(np.prod(range(1, k)))
        for block in part:
            b = sum(conj[i] for i in block)
            term = term*M[len(block) - b, b]
        total += term
    return total

def measure(x, features=FEATURES, power=None):
    """{name: C_pq / power^(p/2)}: power the signal's (default C21, which counts the noise too)"""
    M = _moment_table(x, max(p for _, p, _ in features))
    c21 = float(np.real(M[1, 1] - abs(M[1, 0])**2))
    P = c21 if power is None else power
    return {name: cumulant(M, p, q)/P**(p/2) for name, p, q in features}

def theory(points, features=FEATURES):
    """The normalized cumulants of equiprobable points (unit power)"""
    pts = np.asarray(points, dtype=np.complex128)
    pts = pts/np.sqrt(np.mean(np.abs(pts)**2))
    return measure(pts, features)

def _vector(c):
    """Phase-free feature values: |C40|, C42, C63 real, |C80|, C84 real"""
    return np.array([abs(c["C40"]), c["C42"].real, c["C63"].real, abs(c["C80"]), c["C84"].real])

# Weights: a higher order's estimate scatters more (and its values are larger)
_W = np.array([1.0, 1.0, 0.25, 0.1, 0.05])

class Match:
    """One candidate: name, distance (weighted, of the phase-free cumulants), share (the signal's part of the power that
    fits best), esn0_db (from the share), values (measured, normalized by the signal power), expected (theory)"""
    __slots__ = ("name", "distance", "share", "esn0_db", "values", "expected")
    def __init__(self, **kw):
        for k in self.__slots__:
            setattr(self, k, kw.get(k))

    def __str__(self):
        es = "" if self.esn0_db is None else f", Es/N0 {self.esn0_db:.1f} dB"
        return f"{self.name}: distance {self.distance:.3f}{es}"

    __repr__ = __str__

def candidates():
    """{name: points} of the constellations to compare with"""
    from .. import symbol
    out = {}
    for name, c in (("BPSK", symbol.PSK(2)), ("QPSK", symbol.PSK(4)), ("8PSK", symbol.PSK(8)), ("16PSK", symbol.PSK(16)),
                    ("16QAM", symbol.QAM(16)), ("32QAM", symbol.QAM(32)), ("64QAM", symbol.QAM(64)),
                    ("256QAM", symbol.QAM(256)), ("16APSK", symbol.APSK(16)), ("32APSK", symbol.APSK(32)),
                    ("4ASK", symbol.ASK(4)), ("8ASK", symbol.ASK(8))):
        out[name] = np.asarray(c.points)
    return out

def classify(x, points=None, esn0_db=None, shares=np.linspace(0.05, 1.0, 96)):
    """Candidates by how well the symbols' cumulants match theirs, best first. points: {name: points} (default
    `candidates()`). For each, the signal's share of the power is fitted (noise adds only to C21) unless esn0_db fixes it.

    Families separate on their own (PSK orders, QAM, APSK, ASK, BPSK); orders within QAM or ASK do not without the SNR:
    a larger QAM at a higher SNR has the cumulants of a smaller one with more noise. Give esn0_db (rpps.analysis.quality,
    or the residual of rpps.analysis.cancel) to tell them apart, or fit the constellation (rpps.analysis.modulation)"""
    if esn0_db is not None:
        shares = [1/(1 + 10**(-esn0_db/10))]
    x = np.asarray(x, dtype=np.complex128)
    x = x - np.mean(x)
    M = _moment_table(x, 8)
    c21 = float(np.real(M[1, 1]))
    raw = {name: cumulant(M, p, q) for name, p, q in FEATURES}
    pts = candidates() if points is None else points
    out = []
    for name, P in pts.items():
        t = _vector(theory(P))
        best = None
        for s in shares:
            S = s*c21
            v = _vector({k: raw[k]/S**(p/2) for k, p, _ in FEATURES})
            d = float(np.sqrt(np.sum(_W*(v - t)**2)))
            if best is None or d < best[0]:
                best = (d, s, v)
        d, s, v = best
        es = 10*np.log10(s/(1 - s)) if s < 1 else None
        out.append(Match(name=name, distance=d, share=float(s), esn0_db=es, values=v, expected=t))
    out.sort(key=lambda m: m.distance)
    return out
