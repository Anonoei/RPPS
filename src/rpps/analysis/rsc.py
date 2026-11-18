"""Recursive systematic convolutional codes (turbo constituents) and a turbo code's interleaver, found blind

    c = rp.analysis.rsc.constituent(s, p)            # systematic and parity streams: (g_f, g_b), e.g. 3GPP 15 / 13 octal
    perm = rp.analysis.rsc.turbo_interleaver(S, P2, c.g_f, c.g_b)   # frames: the second encoder's input order

An RSC encoder makes its parity p from the information u = s as p(D) = u(D) g_f(D) / g_b(D): so p g_b + s g_f = 0, a
parity check over the two streams, one label of 2 bits a step. The same Walsh-Hadamard count as rpps.analysis.tcm scores
every pair of polynomials up to the memory at once; the strongest is (g_f, g_b) (a feedforward code: g_b = 1).

A turbo code's second encoder takes the information in another order: its parity p2 has no check with s, which says
that much. With the constituent known, u' = p2 g_b / g_f (an IIR division from the all-zero start of each frame) is the
interleaved information, and over many frames every position of u' matches one position of s bit for bit: the
permutation, one column search per position (Burel; Tixier & Sendrier recover it from noisier frames by BCJR, which this
is not: an error in p2 spreads through the division, so clean frames are needed, or the frames are first decoded).
"""
import numpy as np

from .tcm import fwht, window_counts

def _poly_str(p):
    return " + ".join(("1" if d == 0 else ("D" if d == 1 else f"D^{d}")) for d in range(p.bit_length()) if p >> d & 1)

def octal(p):
    """The octal a standard writes (D^0 the most significant bit, 3GPP's 13 = 1 + D^2 + D^3)"""
    n = p.bit_length()
    return format(int(format(p, f"0{n}b")[::-1], 2), "o")

class Constituent:
    """g_f (feedforward), g_b (feedback): ints, bit d the coefficient of D^d; memory; holds; z"""
    __slots__ = ("g_f", "g_b", "memory", "holds", "z")
    def __init__(self, **kw):
        for k in self.__slots__:
            setattr(self, k, kw.get(k))

    def __str__(self):
        kind = "feedforward" if self.g_b == 1 else "recursive"
        return (f"{kind} code: p = u ({_poly_str(self.g_f)}) / ({_poly_str(self.g_b)}), octal {octal(self.g_f)}/"
                f"{octal(self.g_b)} (D^0 first), memory {self.memory}; check holds {100*self.holds:.1f}% (z {self.z:.0f})")

    __repr__ = __str__

def constituent(s, p, max_memory=6, pfa=1e-6):
    """(g_f, g_b) of the code making parity p from s (aligned streams of equal length), or None"""
    s = np.asarray(s, dtype=np.int64) & 1
    p = np.asarray(p, dtype=np.int64) & 1
    z = s | (p << 1)
    W = max_memory + 1
    counts, N = window_counts(z, 2, W)
    bias = fwht(counts)
    bias[0] = 0
    zs = np.abs(bias)/np.sqrt(N)
    if zs.max() < np.sqrt(2*np.log((1 << (2*W))/pfa)):
        return None
    # The smallest-memory check among the strongest (a multiple of the code's check holds as well)
    top = np.flatnonzero(zs >= 0.95*zs.max())
    best = None
    for v in top:
        v = int(v)
        g = [0, 0]
        for j in range(W):
            d = W - 1 - j # window_counts: bit (W-1-d) m + i is label bit i at delay d
            for i in range(2):
                if v >> (j*2 + i) & 1:
                    g[i] |= 1 << d
        low = min(((q & -q).bit_length() - 1 for q in g if q), default=0)
        g = [q >> low for q in g]
        mem = max(q.bit_length() for q in g) - 1
        if g[1] == 0 or not g[1] & 1:
            continue
        if best is None or mem < best[0]:
            best = (mem, g, float(zs[v]), float((N + bias[v])/(2*N)))
    if best is None:
        return None
    mem, (gf, gb), zv, holds = best
    return Constituent(g_f=gf, g_b=gb, memory=mem, holds=max(holds, 1 - holds), z=zv)

def encode(u, g_f, g_b):
    """Parity of an RSC encoder from the all-zero state: p = u g_f / g_b"""
    u = np.asarray(u, dtype=np.int64) & 1
    m = max(g_f.bit_length(), g_b.bit_length()) - 1
    state = [0]*m # w[t-1] .. w[t-m]
    out = np.zeros(len(u), np.uint8)
    for t, x in enumerate(u):
        w = x # w = u / g_b: w[t] = u[t] + sum_{d>=1} g_b_d w[t-d]
        for d in range(1, m + 1):
            if g_b >> d & 1:
                w ^= state[d - 1]
        y = w if g_f & 1 else 0 # p = w g_f
        for d in range(1, m + 1):
            if g_f >> d & 1:
                y ^= state[d - 1]
        out[t] = y
        state = [w] + state[:-1]
    return out

def invert(p, g_f, g_b):
    """The information an RSC encoder (from the all-zero state) made p from: u = p g_b / g_f (g_f needs a D^0 term)"""
    if not g_f & 1:
        raise ValueError("g_f has no D^0 term: not invertible")
    return encode(p, g_b, g_f) # the same recursion, the roles of the polynomials swapped

def turbo_interleaver(S, P2, g_f, g_b, min_frames=None):
    """The permutation of a turbo code's second encoder: frames S (F, K) of systematic bits and P2 (F, K) of the second
    parity (each frame from the all-zero state). Returns perm (K,): the second encoder's input i is s[perm[i]], and the
    agreement of each match (1: every frame)"""
    S = np.asarray(S, dtype=np.int64) & 1
    P2 = np.asarray(P2, dtype=np.int64) & 1
    F, K = S.shape
    if F < (min_frames or int(np.ceil(np.log2(K))) + 8):
        raise ValueError(f"{F} frames cannot tell {K} positions apart: need about log2(K) + 8")
    U = np.array([invert(p, g_f, g_b) for p in P2]).astype(np.int64)
    # agreement of every (u' position, s position) pair over the frames, as +-1 correlations
    A = (1 - 2*U).T @ (1 - 2*S) # (K, K)
    perm = np.argmax(A, axis=1)
    agree = (A[np.arange(K), perm] + F)/(2*F)
    return perm, agree
