"""Recursive systematic convolutional codes (turbo constituents) and a turbo code's interleaver, found blind

    c = rp.analysis.rsc.constituent(s, p)            # systematic and parity streams: (g_f, g_b), e.g. 3GPP 15 / 13 octal
    perm = rp.analysis.rsc.turbo_interleaver(S, P2, c.g_f, c.g_b)   # frames: the second encoder's input order

An RSC encoder makes its parity p from the information u = s as p(D) = u(D) g_f(D) / g_b(D): so p g_b + s g_f = 0, a
parity check over the two streams, one label of 2 bits a step. The same Walsh-Hadamard count as rpps.analysis.tcm scores
every pair of polynomials up to the memory at once; the strongest is (g_f, g_b) (a feedforward code: g_b = 1).

A turbo code's second encoder takes the information in another order: its parity p2 has no check with s, which says
that much. With the constituent known, u' = p2 g_b / g_f (an IIR division from the all-zero start of each frame) is the
interleaved information, and over many frames every position of u' matches one position of s bit for bit: the
permutation, one column search per position (Burel). An error in p2 spreads through the division, so that needs clean
frames.

From noisy frames (soft values), `turbo_interleaver_soft` uses the constituent's check instead of its inverse:
p2 g_b + u' g_f = 0 holds at every step t over p2[t - d] and u'[t - d] = s[perm[t - d]], d up to the memory, so
with perm[t - 1] ... perm[t - m] found, each candidate j for perm[t] is scored over all frames at once by the check's
soft value: sum over frames of tanh(L_s[j]/2) times the product of tanh(L/2) of the check's other bits (one matrix-
vector product a position; positions taken are excluded). A wrong pick spoils the next m checks, so a repair pass then
scores every position by all m + 1 checks it is in, with every other position as found, and reassigns them greedily
from the most confident (a sequential search with error correction in the spirit of Tixier & Sendrier, without BCJR).
The forward search keeps a small beam, so one wrong pick doesn't take every later one with it.
The check's weight w = |g_f| + |g_b| bits sets the noise it takes: its soft value falls as (1 - 2 BER)^w, so the frames
needed grow as (1 - 2 BER)^(-2w) log K (3GPP's w = 6: about 100 frames of 1000 bits at 4% bit errors).
"""
import numpy as np

from ..math.gf2 import fwht
from .tcm import window_counts

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

def _taps(p):
    return [d for d in range(p.bit_length()) if p >> d & 1]

def turbo_interleaver_soft(Ls, Lp2, g_f, g_b, beam=4, expand=3, repair=2):
    """The second encoder's permutation from noisy frames: Ls (F, K) LLRs (log P(0)/P(1)) of the systematic bits, Lp2
    (F, K) of the second parity (hard bits work as +-large values). g_f needs a D^0 term. beam: partial permutations
    kept, each extended by its `expand` best candidates a step; repair: passes of the repair. Returns (perm (K,): input i
    of the second encoder is s[perm[i]], confidence (K,): each position's best check sum over its second best's;
    near 1 is a guess)"""
    Ts = np.tanh(np.clip(np.asarray(Ls, dtype=np.float64), -30, 30)/2)
    Tp = np.tanh(np.clip(np.asarray(Lp2, dtype=np.float64), -30, 30)/2)
    F, K = Ts.shape
    if not g_f & 1:
        raise ValueError("g_f has no D^0 term")
    fd, bd = _taps(g_f), _taps(g_b)

    def known(t, perm, skip=None):
        """(F,): the soft value of the check at step t without s[perm[skip]] (skip: the delay left out; None: all in)"""
        v = np.ones(F)
        for d in bd:
            if t - d >= 0:
                v *= Tp[:, t - d]
        for d in fd:
            if d == skip or t - d < 0:
                continue
            if perm[t - d] < 0:
                return None
            v *= Ts[:, perm[t - d]]
        return v

    # forward: a beam of partial permutations, each extended by its best candidates, ranked by their summed check
    # values over the spread a check has on random candidates (so a wrong pick that spoils later checks falls back)
    beams = [(0.0, np.full(K, -1, np.int64), np.zeros(K, bool))]
    for t in range(K):
        ext = []
        for total, pm, used in beams:
            v = known(t, pm, skip=0)
            sc = v @ Ts
            spread = np.sqrt(np.sum(v**2)) + 1e-12
            sc[used] = -np.inf
            for j in np.argpartition(-sc, min(expand, K - 1))[:expand]:
                if np.isfinite(sc[j]):
                    ext.append((total + sc[j]/spread, pm, used, int(j)))
        ext.sort(key=lambda e: -e[0])
        beams = []
        for total, pm, used, j in ext[:beam]:
            pm, used = pm.copy(), used.copy()
            pm[t], used[j] = j, True
            beams.append((total, pm, used))
    perm = beams[0][1]

    def scores(perm):
        """(K, K): position t, candidate j: the sum over the checks t + d (d in g_f) of the check without s[perm[t]]"""
        S = np.zeros((K, K))
        for t in range(K):
            for d in fd:
                if t + d < K:
                    v = known(t + d, perm, skip=d)
                    S[t] += v @ Ts
        return S

    for _ in range(repair):
        S = scores(perm)
        top2 = np.sort(S, axis=1)[:, -2:]
        conf = top2[:, 1] - top2[:, 0]
        new = np.full(K, -1, np.int64)
        taken = np.zeros(K, bool)
        for t in np.argsort(-conf):
            row = np.where(taken, -np.inf, S[t])
            j = int(np.argmax(row))
            new[t], taken[j] = j, True
        if np.array_equal(new, perm):
            break
        perm = new
    S = scores(perm)
    best = S[np.arange(K), perm]
    S2 = S.copy()
    S2[np.arange(K), perm] = -np.inf
    second = S2.max(axis=1)
    conf = best/np.maximum(np.abs(second), 1e-12)
    return perm, conf
