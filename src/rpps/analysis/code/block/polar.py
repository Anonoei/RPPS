"""A polar code's frozen positions, blind: which of the u bits are always 0, and which construction froze them

    r = rp.analysis.code.block.polar.frozen(llr)       # frames (W, N) of LLRs, N a power of 2, aligned: the column test
    r = rp.analysis.code.block.polar.recognize(llr)    # at low SNR: successive cancellation, a test per position
    r = rp.analysis.code.block.polar.search(llr)       # and the construction (polarization weight, Bhattacharyya) and K it is
    r.frozen, r.k

The code itself (encoding, SC and list decoding) is rpps.code.types.polar. A polar codeword is x = u G with G = F^(x n), F = [[1, 0], [1, 1]]; G is its own inverse, so u = x G and u_i is the parity
of the x bits in column i of G. The frozen positions of u are 0 in every frame; the information positions are 0 half the
time.

`frozen` (the column test): each u_i's LLR is the tanh-product of its column's x LLRs (as a parity check's); averaged
over frames it is high at frozen positions and near zero at information ones (the hypothesis test of arXiv
2606.17705). A heavy column sees many x errors, and the early positions a real code freezes have the heaviest (column 0
is the parity of all N bits): a (64, 32) frozen set is found at Es/N0 6 dB from 400 frames, but a rate-1/2 code built
by polarization weight is not found at +2 dB.

`recognize` (blind successive cancellation): the SC decoder runs over every frame at once, and at each position, in
decoding order, every frame's SC LLR of u_i given the u before it (frozen ones 0, information ones as decided) is a
much stronger statistic than the bare column, because the positions before it are taken out. Per position the
generalised likelihood ratio of "u_i = 0 always" against "u_i uniform", sum over frames of log2(2 / (1 + e^-L)), is
W times the synthetic channel's capacity when frozen and at most 0 when not; a position is information only when it is
significantly negative (a channel too poor to tell is frozen, as every construction does), and the partial order every
construction keeps (i at least as reliable as each j whose bits are a subset of its own) carries an information
verdict upward. (In the spirit of blind SC/SCL recognition, arXiv 2605.13331, without the list.)

`search` names the construction: each reliability order (polarization weight, the beta-expansion 3GPP NR's sequence
follows; Bhattacharyya parameters at a few design SNRs) with the K whose least reliable N - K positions are nearest
the blind set; within a few positions, the construction's set is returned (which corrects the blind test's errors).

Measured, rate 1/2 built by polarization weight, 200 frames: (128, 64), (256, 128), (512, 256) found exactly down to
Es/N0 -2 dB by `recognize`; `search` exact at -3 dB and within a few positions at -4 dB; `frozen` fails at +2 dB.
"""
import numpy as np

from ....code.types.polar import transform, sc, pw_order, bhattacharyya_order

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

def evidence(L):
    """Per frame, log2 of how much likelier an SC LLR L is with u = 0 than with u uniform: log2(2 / (1 + e^-L)). Its sum
    over frames is positive (W times the synthetic channel's capacity) at a frozen position, at most 0 at an information
    one: the generalised likelihood ratio of the two"""
    L = np.asarray(L, dtype=np.float64)
    return 1.0 - np.logaddexp(0.0, -L)/np.log(2)

class Recognition:
    """Result of `recognize`: frozen (positions), k, evidence (N,): each position's summed evidence (bits; > 0 frozen),
    construction (the reliability order it fits, with `search`), score"""
    __slots__ = ("frozen", "k", "evidence", "construction", "score")
    def __init__(self, **kw):
        for k in self.__slots__:
            setattr(self, k, kw.get(k))

    def __str__(self):
        N = len(self.evidence)
        how = f", the {self.construction} construction" if self.construction else ""
        return f"polar ({N}, {self.k}){how}: {len(self.frozen)} frozen positions"

    __repr__ = __str__

def recognize(llr, z=3.0, monotone=True):
    """Frozen positions of aligned polar frames (W, N) of LLRs at low SNR: blind SC (see the module). Position by
    position in decoding order, every frame's SC LLR of u_i (given the u before it: frozen ones 0, information ones as
    decided) gives its `evidence`; the position is information when the evidence is significantly negative (its sum
    under -z of its spread), else frozen: a position too poor to tell either way is a channel any construction freezes.
    monotone: every polar construction keeps the partial order where i is at least as reliable as each j whose set
    bits are a subset of its own (j & i == j): so a position over an information one is information too"""
    L = np.asarray(llr, dtype=np.float64)
    W, N = L.shape
    ev = np.zeros(N)
    info = np.zeros(N, bool)

    def decide(i, Li):
        e = evidence(Li)
        ev[i] = float(e.sum())
        is_info = ev[i] < -z*np.sqrt(float(np.sum(e**2)) + 1e-12)
        if monotone and not is_info:
            js = np.flatnonzero(info[:i])
            if np.any((js & i) == js):
                is_info = True
        info[i] = is_info
        return (Li < 0).astype(np.uint8) if is_info else np.zeros(W, np.uint8)

    sc(L, decide)
    fz = np.flatnonzero(~info)
    return Recognition(frozen=fz, k=N - len(fz), evidence=ev, construction=None, score=float(ev[fz].sum()))

def search(llr, orders=None, tolerance=0.04, **kw):
    """Which construction (and K) the frozen set of frames (W, N) of LLRs is: `recognize` finds the set blind, then each
    reliability order (default polarization weight, and Bhattacharyya at -2 / 0 / 2 / 4 dB) is matched by its K whose
    N - K least reliable positions differ from it in the fewest places. Within tolerance x N places, the construction's
    set is returned (the few positions a noisy blind test got wrong come right), named; else the blind set, unnamed.
    kw: recognize's. Returns Recognition (score: the places the construction differs from the blind set)"""
    L = np.asarray(llr, dtype=np.float64)
    W, N = L.shape
    blind = recognize(L, **kw)
    if orders is None:
        orders = {"polarization weight": pw_order(N)}
        for d in (-2, 0, 2, 4):
            orders[f"Bhattacharyya {d:+d} dB"] = bhattacharyya_order(N, d)
    fz = np.zeros(N, bool)
    fz[blind.frozen] = True
    best = None
    for name, order in orders.items():
        order = np.asarray(order)
        # freezing the first m of the order: differences = blind-frozen positions left out + others put in
        inc = np.where(fz[order], -1, 1) # each position added changes the count by this
        diff = int(fz.sum()) + np.concatenate(([0], np.cumsum(inc)))
        m = int(np.argmin(diff[1:N])) + 1
        if best is None or diff[m] < best[0]:
            best = (int(diff[m]), name, np.sort(order[:m]))
    d, name, frozen = best
    if d > tolerance*N:
        blind.score = float(d)
        return blind
    return Recognition(frozen=frozen, k=N - len(frozen), evidence=blind.evidence, construction=name, score=float(d))
