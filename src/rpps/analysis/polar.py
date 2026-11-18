"""A polar code's frozen positions, blind: which of the u bits are always 0

    r = rp.analysis.polar.frozen(llr)       # frames (W, N) of LLRs, N a power of 2, aligned
    r.frozen, r.k                           # frozen positions, K = N - len(frozen)

A polar codeword is x = u G with G = F^(x n), F = [[1, 0], [1, 1]]; G is its own inverse, so u = x G and u_i is the parity
of the x bits in column i of G. The frozen positions of u are 0 in every frame; the information positions are 0 half the
time. From LLRs, each u_i's LLR is the tanh-product of its column's x LLRs (as a parity check's); averaged over frames
(as the expected 0-ness) it is high at frozen positions and near zero at information ones: a test per position (the
hypothesis test of arXiv 2606.17705, without the list search of blind SCL recognition, arXiv 2605.13331, that
copes with more noise). A position whose column is heavy sees many x errors; those are the early positions, the ones a
real code freezes (column 0 is the parity of all N bits): a (64, 32) Reed-Muller-like frozen set is found exactly at
Es/N0 6 dB from 400 frames.
"""
import numpy as np

def transform(N):
    """G = F^(x n) for N = 2^n, uint8"""
    n = int(np.log2(N))
    if 2**n != N:
        raise ValueError("N is a power of 2")
    G = np.array([[1]], np.uint8)
    F = np.array([[1, 0], [1, 1]], np.uint8)
    for _ in range(n):
        G = np.kron(G, F)
    return G

class Frozen:
    """Result of `frozen`: frozen (positions), k (information bits), score (each position's mean tanh(L/2) of u_i: 1 for
    a frozen position on clean frames, 0 for information), threshold"""
    __slots__ = ("frozen", "k", "score", "threshold")
    def __init__(self, **kw):
        for k in self.__slots__:
            setattr(self, k, kw.get(k))

    def __str__(self):
        N = len(self.score)
        return f"polar ({N}, {self.k}): {len(self.frozen)} frozen positions"

    __repr__ = __str__

def frozen(llr, threshold=None):
    """Frozen positions from frames (W, N) of LLRs (log P(0)/P(1)) of aligned polar codewords (see the module).
    threshold: the score above which a position is frozen (default halfway between the information positions' (about 0)
    and the frozen ones', from the scores' two clusters)"""
    L = np.asarray(llr, dtype=np.float64)
    W, N = L.shape
    G = transform(N)
    t = np.tanh(np.clip(L, -30, 30)/2) # (W, N): E[(-1)^x] per bit
    # u_i = parity of x over column i: tanh(L_u/2) = prod over the column of tanh(L_x/2); in logs for the sign and size
    mag = np.log(np.maximum(np.abs(t), 1e-300)) @ G.astype(np.float64) # (W, N)
    neg = ((t < 0).astype(np.int64) @ G.astype(np.int64)) % 2
    score = np.mean(np.where(neg == 1, -1.0, 1.0)*np.exp(mag), axis=0)
    if threshold is None:
        s = np.sort(score)
        gaps = np.diff(s)
        i = int(np.argmax(gaps)) # the widest gap between the two clusters
        threshold = (s[i] + s[i + 1])/2
    fz = np.flatnonzero(score > threshold)
    return Frozen(frozen=fz, k=N - len(fz), score=score, threshold=float(threshold))
