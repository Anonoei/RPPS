"""Blind LDPC and algebraic block codes, past what one search finds: staircase (IRA) codes, progressive reconstruction, roots

    H, holds = rp.analysis.code.block.ldpc.staircase(words, k)          # dual-diagonal parity (DVB-S2, 802.11n, 802.16e, 5G NR)
    H, holds, rounds = rp.analysis.code.block.ldpc.progressive(llr)     # checks found, frames cleaned with them, more found, ...
    r = rp.analysis.code.block.ldpc.roots(words)                        # a binary cyclic code's roots: BCH and its distance
    z = rp.analysis.code.block.ldpc.circulant_size(words, k)            # quasi-cyclic: the circulant size
    V, c, holds = rp.analysis.code.block.ldpc.qc_generator(words, k, z) # its generator, (k/z) x ((n-k)/z) circulants
    H, rank = rp.analysis.code.block.ldpc.qc_checks(V, z)               # and sparse checks from it

staircase: an IRA code's parity part is dual-diagonal: parity bit j is parity bit j-1 plus a sum of information bits, the
row of A in H = [A | staircase]. With k information bits first, d_j = p_j + p_{j-1} is a linear function of them, the same
for every word: over k + some words, one GF(2) elimination of the information bits solves every row at once (no search
for sparse dual words: Valembois, Cluzeau, Canteaut-Chabaud would look for each). Noisy bits: the most reliable words are
used, several subsets, and each row is kept by its vote and checked on words it was not solved from.

progressive: the checks one search finds (rpps.analysis.code.block.linear.sparse_checks: information sets) decode the frames (min-sum
on the partial H), the decoded frames have fewer errors, and the next search finds checks the first could not, until no
new ones come (Cluzeau 2006, block code reconstruction with iterative decoding; the ISIT 2024 progressive QC-LDPC
reconstruction reaches ~100x the bit error rate of one search).

quasi-cyclic (qc_generator, qc_checks): a QC code's generator is circulants, so each word gives z equations at once and a
few dozen words solve it (k/z + 9 of them, where a dense search for dual words needs over k); the sparse checks are then
found in the generator itself, as sums of a few shifted parity blocks whose information dependences nearly cancel (an
LDPC code's dense generator hides them from a word-level search unless words are many).

roots: a binary cyclic code of length n | 2^m - 1 is its generator's roots: every codeword c has c(beta^j) = 0 at them
(the Vandermonde parity checks). Evaluated at every beta^j, words of the code are zero at the roots and a fraction 2^-m
elsewhere; a word with an error is not zero there either, so the zero fraction is the clean words' share, which stands far
over 2^-m at a few percent of clean words. Consecutive roots give a BCH code's designed distance.
"""
import numpy as np

from ....math.gf import GF
from ....math.gf2 import rref, rank, to_checks
from ....code.types.ldpc import ldpc as _ldpc
from ....logger import Logger
from .linear import sparse_checks

# Primitive polynomials (x^m term included) for GF(2^m)
PRIMITIVE = {2: 0b111, 3: 0b1011, 4: 0b10011, 5: 0b100101, 6: 0b1000011, 7: 0b10001001, 8: 0x11D, 9: 0x211, 10: 0x409,
             11: 0x805, 12: 0x1053, 13: 0x201B, 14: 0x4443, 15: 0x8003, 16: 0x1100B}

def _expected_errors(rel):
    return (1.0/(1.0 + np.exp(np.minimum(np.asarray(rel, dtype=float), 50.0)))).sum(axis=1)

def _solve(U, D):
    """X with U X = D over GF(2) when U (r x k) has rank k: (X (k x c), ok)"""
    r, k = U.shape
    A = np.concatenate((U, D), axis=1).astype(np.uint8)
    R, piv = rref(A)
    piv = [p for p in piv if p < k]
    if len(piv) < k:
        return None, False
    return R[:k, k:], True

def staircase(words, k, rel=None, trials=16, verify=256, seed=0, extra=()):
    """Rows of an IRA code's H from aligned words (W, n) whose first k positions are the information bits and the rest
    the parity bits in accumulator order. extra: parity positions (indices into the parity part) that some rows also
    involve (802.11n's first parity column is weight 3): they join the information bits as regressors.
    Returns (H (n-k, n) uint8, fraction of held-out words each row holds on)"""
    words = np.asarray(words, dtype=np.uint8)
    W, n = words.shape
    m = n - k
    P = words[:, k:]
    D = P.copy()
    D[:, 1:] ^= P[:, :-1] # d_j = p_j + p_{j-1}
    cols = list(range(k)) + [k + e for e in extra]
    U = words[:, cols]
    order = np.argsort(_expected_errors(rel), kind="stable") if rel is not None else np.arange(W)
    use = len(cols) + 24
    if W < use + 32:
        raise ValueError(f"need about {use + 32} words for k = {k}, have {W}")
    pool, test = order[:max(use, W - verify)], order[max(use, W - verify):]
    rng = np.random.default_rng(seed)
    # Each trial solves every row; per row, the solution holding on the most held-out words is kept (an error in a word
    # used spoils a trial's rows, not all trials')
    T = test[:verify]
    best = np.full(m, -1.0)
    X = np.zeros((len(cols), m), np.uint8)
    for t in range(trials):
        sub = pool[:use] if t == 0 else rng.choice(pool[:min(len(pool), 3*use)], use, replace=False)
        Xt, ok = _solve(U[sub], D[sub])
        if not ok:
            continue
        held = 1.0 - ((U[T].astype(np.int32) @ Xt.astype(np.int32) + D[T]) % 2).mean(axis=0)
        better = held > best
        X[:, better], best[better] = Xt[:, better], held[better]
    if np.all(best < 0):
        raise ValueError("the information bits do not have full rank over the words: not k information bits first")
    H = np.zeros((m, n), np.uint8)
    for j in range(m):
        H[j, np.array(cols)[X[:, j] == 1]] ^= 1
        H[j, k + j] ^= 1
        if j:
            H[j, k + j - 1] ^= 1
    holds = 1.0 - ((words[test].astype(np.int32) @ H.T.astype(np.int32)) % 2).mean(axis=0)
    return H, holds

def _qc_system(U, z):
    """The equations of a quasi-cyclic generator: for words U (W, kb, z) of information blocks, row (w, o) has the
    coefficients of V[b, s] (b * z + s: parity bit o of a block is sum over b, s of u[b, (o - s) % z] V[b, s]) and a one
    for the block's constant c[o]. (W * z, kb * z + z)"""
    W, kb, _ = U.shape
    idx = (np.arange(z)[:, None] - np.arange(z)[None, :]) % z # (o, s) -> t
    A = U[:, :, idx] # (W, kb, o, s)
    A = A.transpose(0, 2, 1, 3).reshape(W*z, kb*z)
    return np.concatenate((A, np.tile(np.eye(z, dtype=np.uint8), (W, 1))), axis=1)

def qc_generator(words, k, z, rel=None, trials=8, verify=256, seed=0):
    """The generator of a quasi-cyclic code (circulant size z) from aligned words (W, n) with the k information bits
    first: parity bit o of parity block J is sum over information blocks b and s of u[b, (o - s) % z] V[b, J, s], plus
    c[J z + o] (zero for a linear code; a coset's constant otherwise). A shift of every information block by one shifts
    every parity block by one, so the k x (n - k) generator is (k/z) x ((n - k)/z) circulants, each one row of z bits:
    each word gives z equations in the k + z unknowns of a parity block, and the coefficients are the same for every
    parity block, so one elimination over about k/z + 9 words solves them all. Noisy bits: the most reliable words, several
    subsets, each parity block's solution kept by how many held-out equations it satisfies.
    Returns (V (k/z, (n-k)/z, z) uint8, c (n - k,) uint8, holds ((n-k)/z,): the fraction of held-out parity bits each
    block's solution predicts)"""
    words = np.asarray(words, dtype=np.uint8)
    W, n = words.shape
    if k % z or (n - k) % z:
        raise ValueError(f"k = {k} and n - k = {n - k} are not multiples of z = {z}")
    kb, mb = k//z, (n - k)//z
    order = np.argsort(_expected_errors(rel), kind="stable") if rel is not None else np.arange(W)
    use = kb + 9
    if W < use + 8:
        raise ValueError(f"need about {use + 8} words for k = {k}, z = {z}, have {W}")
    pool, test = order[:max(use, W - verify)], order[max(use, W - verify):]
    U = words[:, :k].reshape(W, kb, z)
    Y = words[:, k:].reshape(W, mb, z).transpose(0, 2, 1).reshape(W*z, mb) # row (w, o), column J
    rng = np.random.default_rng(seed)
    At = _qc_system(U[test], z)
    Yt = Y.reshape(W, z, mb)[test].reshape(-1, mb)
    best = np.full(mb, -1.0)
    X = np.zeros((k + z, mb), np.uint8)
    for t in range(trials):
        sub = pool[:use] if t == 0 else rng.choice(pool[:min(len(pool), 3*use)], use, replace=False)
        Xt, ok = _solve(_qc_system(U[sub], z), Y.reshape(W, z, mb)[sub].reshape(-1, mb))
        if not ok:
            continue
        held = 1.0 - ((At.astype(np.int32) @ Xt.astype(np.int32) + Yt) % 2).mean(axis=0)
        better = held > best
        X[:, better], best[better] = Xt[:, better], held[better]
    if np.all(best < 0):
        raise ValueError("the information blocks do not have full rank over the words: not k information bits first, "
                         "or too few different words")
    V = X[:k].reshape(kb, z, mb).transpose(0, 2, 1).copy()
    c = X[k:].T.reshape(-1).copy()
    return V, c, best

def circulant_size(words, k, sizes=None, rel=None, hold=0.95, **kw):
    """The circulant size of a quasi-cyclic code: the smallest z (of `sizes`, default every common divisor of k and n - k
    from 2) whose qc_generator predicts held-out words (every parity block's fraction >= hold), or None. kw: its"""
    n = np.asarray(words).shape[1]
    if sizes is None:
        g = np.gcd(k, n - k)
        sizes = [z for z in range(2, g + 1) if g % z == 0]
    for z in sizes:
        try:
            _, _, holds = qc_generator(words, k, z, rel=rel, **kw)
        except ValueError:
            continue
        if np.all(holds >= hold):
            return z
    return None

def _qc_dependence(V, z):
    """D[J, o]: the information bits parity bit (J, o) depends on, as a k-bit row"""
    kb, mb, _ = V.shape
    t = np.arange(z)
    D = np.zeros((mb, z, kb*z), np.uint8)
    for o in range(z):
        D[:, o] = V[:, :, (o - t) % z].transpose(1, 0, 2).reshape(mb, kb*z)
    return D

def qc_checks(V, z, max_weight=8, quads=True, reps=12, seed=0):
    """Sparse parity checks of the quasi-cyclic code qc_generator found: H ((n - k), n) over [information k | parity
    n - k], full rank when they are found, lightest block rows first.

    A block row of checks is a sum of shifted parity blocks whose information dependences cancel but for a few bits:
    x^a p_J1 + x^b p_J2 (+ x^c p_J3 (+ x^d p_J4)) = a sparse sum of information bits, for every shift together (z checks).
    LDPC codes have them with few terms (the generator, H_p^-1 H_u, is dense: a search for sparse dual words in the words
    themselves needs many), so: every pair and triple of parity blocks over all shifts, and quadruples by meeting in the
    middle (pairs whose dependences nearly agree: locality-sensitive hashing on `reps` random 40-bit projections), each
    kept when its information part has at most `max_weight` bits and it adds rank. Returns (H, rank)"""
    V = np.asarray(V, dtype=np.uint8)
    kb, mb, _ = V.shape
    k, m = kb*z, mb*z
    D = _qc_dependence(V, z)
    pad = (-k) % 64
    Dp = np.packbits(np.concatenate((D, np.zeros((mb, z, pad), np.uint8)), axis=2), axis=2).view(np.uint64)
    def weight(x):
        return np.unpackbits(np.ascontiguousarray(x).view(np.uint8), axis=-1).sum(axis=-1)
    rels = [] # (weight, ((J, shift), ...))
    for J1 in range(mb):
        for J2 in range(J1 + 1, mb):
            w = weight(Dp[J1, 0][None] ^ Dp[J2])
            for o in np.flatnonzero(w <= max_weight):
                rels.append((int(w[o]), ((J1, 0), (J2, int(o)))))
    for J3 in range(mb):
        for J1 in range(mb):
            for J2 in range(J1 + 1, mb):
                if J3 in (J1, J2):
                    continue
                w = weight(Dp[J3, 0][None, None] ^ Dp[J1][:, None] ^ Dp[J2][None, :])
                for a, b in zip(*np.nonzero(w <= max_weight)):
                    rels.append((int(w[a, b]), ((J3, 0), (J1, int(a)), (J2, int(b)))))
    if quads and mb >= 4:
        rng = np.random.default_rng(seed)
        seen = set()
        for _ in range(reps):
            proj = rng.choice(k, min(40, k), replace=False)
            P = D[:, :, proj] # (mb, z, 40)
            buckets = {}
            for J1 in range(mb):
                for J2 in range(J1 + 1, mb):
                    keys = np.packbits(P[J1][:, None, :] ^ P[J2][None, :, :], axis=2)
                    for a in range(z):
                        for b in range(z):
                            buckets.setdefault(keys[a, b].tobytes(), []).append((J1, a, J2, b))
            for lst in buckets.values():
                if len(lst) < 2 or len(lst) > 64:
                    continue
                for i in range(len(lst)):
                    for j in range(i + 1, len(lst)):
                        p, q = lst[i], lst[j]
                        if len({p[0], p[2], q[0], q[2]}) < 4:
                            continue
                        terms = tuple(sorted(((p[0], p[1]), (p[2], p[3]), (q[0], q[1]), (q[2], q[3]))))
                        base = tuple((J, (s - terms[0][1]) % z) for J, s in terms) # one per block row
                        if base in seen:
                            continue
                        seen.add(base)
                        w = int(weight(Dp[p[0], p[1]] ^ Dp[p[2], p[3]] ^ Dp[q[0], q[1]] ^ Dp[q[2], q[3]]))
                        if w <= max_weight:
                            rels.append((w, base))
    rels.sort(key=lambda r: (r[0] + len(r[1]), r[1]))
    basis, rows = {}, []
    for _, terms in rels:
        for kk in range(z):
            h = np.zeros(k + m, np.uint8)
            v = np.zeros(k, np.uint8)
            for J, s in terms:
                o = (s + kk) % z
                h[k + J*z + o] ^= 1
                v ^= D[J, o]
            h[:k] = v
            x = int.from_bytes(np.packbits(h).tobytes(), "big") # a new direction? (elimination on leading bits)
            while x:
                lead = x.bit_length()
                if lead not in basis:
                    basis[lead] = x
                    rows.append(h)
                    break
                x ^= basis[lead]
        if len(rows) >= m:
            break
    H = np.array(rows, dtype=np.uint8) if rows else np.zeros((0, k + m), np.uint8)
    return H, len(rows)

def posterior(H, llr, iters=20, alpha=0.8):
    """A-posteriori LLRs of words (B, n) through the checks H (c, n) by normalized min-sum, flooding (all checks at once):
    what the checks known so far say about each bit (rpps.code.types.ldpc.flood)"""
    H = np.asarray(H, dtype=bool)
    code = _ldpc(Logger(), H.shape[1], to_checks(H), k=0, systematic_pos=[], iters=int(iters), alpha=float(alpha))
    return code.flood(llr)

def _keep(H, limit):
    """The lightest checks of H that span what it spans, then lighter others up to `limit`"""
    H = H[np.argsort(H.sum(axis=1), kind="stable")]
    basis, rows, extra = [], [], []
    for h in H: # greedy: a check is new if it adds to the rank of the lighter ones kept (bit-packed elimination)
        v = int("".join(map(str, h)), 2)
        for bv in basis:
            v = min(v, v ^ bv)
        if v:
            basis.append(v)
            rows.append(h)
        elif len(extra) < limit:
            extra.append(h)
    out = rows + extra[:max(0, limit - len(rows))]
    return np.array(out, dtype=np.uint8), len(rows)

def progressive(llr, rounds=8, iters=20, max_checks=None, **kw):
    """Checks of aligned frames (W, n) of LLRs (log P(0)/P(1)), found again and again on frames cleaned by the checks
    found so far (see the module). kw: rpps.analysis.code.block.linear.sparse_checks's. Stops when a round adds no rank.
    Returns (H (c, n): a basis of the checks found, lightest first, then more light ones; holds on the original hard
    bits; [rank after each round])"""
    llr = np.asarray(llr, dtype=np.float64)
    n = llr.shape[1]
    max_checks = max_checks or 2*n
    kw.setdefault("trials", 60)
    words0 = (llr < 0).astype(np.uint8)
    H = np.zeros((0, n), np.uint8)
    history = []
    cur = llr
    for r in range(rounds):
        new, _ = sparse_checks((cur < 0).astype(np.uint8), rel=np.abs(cur), seed=r, **kw)
        if len(new):
            H, rk = _keep(np.concatenate((H, new)), max_checks)
        else:
            rk = history[-1] if history else 0
        grew = not history or rk > history[-1]
        history.append(rk)
        if not len(H) or (not grew and r > 0):
            break
        cur = posterior(H, llr, iters) # always from the channel's LLRs: the checks only add what they know
    holds = 1.0 - ((words0.astype(np.int32) @ H.T.astype(np.int32)) % 2).mean(axis=0) if len(H) else np.zeros(0)
    return H, holds, history

class Roots:
    """Result of `roots`: n, m, roots (exponents j of beta = alpha^((2^m - 1)/n), sorted), zero (fraction of words zero at
    each root), generator (int, bit i the coefficient of x^i), k (n - degree), designed (BCH designed distance: 1 + the
    longest run of consecutive roots), clean (estimated share of words without an error)"""
    __slots__ = ("n", "m", "roots", "zero", "generator", "k", "designed", "clean")
    def __init__(self, **kw):
        for k in self.__slots__:
            setattr(self, k, kw.get(k))

    def __str__(self):
        if not self.roots:
            return f"no roots: not a cyclic code of length {self.n} (or too few clean words)"
        return (f"cyclic ({self.n}, {self.k}), generator 0x{self.generator:x}, roots {self.roots}, BCH designed distance "
                f"{self.designed}; {100*self.clean:.0f}% of words clean")

    __repr__ = __str__

def _minimal(gf, e, n_field):
    """Minimal polynomial over GF(2) of alpha^e (int, bit i the x^i coefficient), and its cyclotomic coset"""
    coset = []
    x = e % n_field
    while x not in coset:
        coset.append(x)
        x = (2*x) % n_field
    poly = np.array([1], dtype=np.int64) # highest degree first
    for c in coset:
        poly = gf.polymul(poly, np.array([1, gf.pow_alpha(c)]))
    bits = poly[::-1] # lowest first
    if np.any(bits > 1):
        raise ValueError("minimal polynomial not binary")
    return int(sum(int(b) << i for i, b in enumerate(bits))), coset

def roots(words, m=None, z_min=8.0, reverse=False):
    """Roots of a binary cyclic code from aligned words (W, n), position i the coefficient of x^i (reverse: of
    x^(n-1-i)). m: the field (default the smallest with n | 2^m - 1)"""
    words = np.asarray(words, dtype=np.int64)
    if reverse:
        words = words[:, ::-1]
    W, n = words.shape
    if m is None:
        m = next((mm for mm in range(2, 17) if (2**mm - 1) % n == 0), None)
        if m is None:
            raise ValueError(f"no GF(2^m), m <= 16, has an n = {n}th root of unity")
    gf = GF(m, PRIMITIVE[m])
    N = 2**m - 1
    step = N//n
    i = np.arange(n)
    zero = np.zeros(n)
    for j in range(n):
        zero[j] = np.mean(gf.evaluate(words, i*j*step) == 0)
    # c(beta^j) of a random word lies in the subfield its conjugates span, GF(2^|coset|): zero 2^-|coset| of the time
    base = np.array([2.0**-len(_minimal(gf, j*step, N)[1]) for j in range(n)])
    z = (zero - base)/np.sqrt(base*(1 - base)/W)
    found = [j for j in range(n) if z[j] > z_min]
    # A code's roots come in cyclotomic cosets: keep only whole cosets
    keep = set()
    for j in found:
        _, coset = _minimal(gf, j*step, N)
        js = {c//step for c in coset}
        if js <= set(found):
            keep |= js
    rts = sorted(keep)
    g = 1
    done = set()
    for j in rts:
        if j in done:
            continue
        p, coset = _minimal(gf, j*step, N)
        done |= {c//step for c in coset}
        out = 0 # g * p over GF(2)
        a, b = g, p
        while b:
            if b & 1:
                out ^= a
            a <<= 1
            b >>= 1
        g = out
    run = best = 0
    for j in range(1, n + 1):
        run = run + 1 if (j % n) in keep else 0
        best = max(best, run)
    clean = float(np.median(zero[rts] - base[rts])) if rts else 0.0
    return Roots(n=n, m=m, roots=rts, zero=zero, generator=g if rts else 0, k=n - (g.bit_length() - 1) if rts else n,
                 designed=best + 1, clean=clean)

# ---- Quasi-cyclic codes: structure exploited, so frames needed scale with k / Z, not k ----

def _qc_rows(x, Z, G):
    """For one frame's bits x (G Z of them): the (Z, G Z) block of the circulant system: row j, column (g, t) is
    x[Z g + (j - t) mod Z]"""
    idx = (np.arange(Z)[:, None] - np.arange(Z)[None, :]) % Z # (j, t)
    X = np.asarray(x[:G*Z], dtype=np.uint8).reshape(G, Z)
    return X[:, idx].transpose(1, 0, 2).reshape(Z, G*Z) # (j, g, t)

class QCStaircase:
    """Result of `qc_staircase`: table (per information group, the parity addresses x: DVB-S2's address table), checks
    (positions per check, as rpps.code.types.ira.ira_checks gives), holds (per row class, on held-out frames), Z"""
    __slots__ = ("table", "checks", "holds", "Z", "n", "k")
    def __init__(self, **kw):
        for k in self.__slots__:
            setattr(self, k, kw.get(k))

    def __str__(self):
        return (f"QC-IRA ({self.n}, {self.k}), circulants of {self.Z}: {sum(len(t) for t in self.table)} addresses over "
                f"{len(self.table)} groups; rows hold {100*float(np.min(self.holds)):.1f}% at worst")

    __repr__ = __str__

def qc_staircase(words, k, Z=360, rel=None, extra=12, trials=3, seed=0):
    """A DVB-S2-style IRA code (information bits in groups of Z, each adding to parity rows x + m q, parity accumulated)
    from aligned codewords (W, n), information first: its address table, from about k/Z + `extra` clean words (the most
    reliable ones, by `rel` |LLR|; `trials` subsets, each row class keeping the solution that holds best)"""
    words = np.asarray(words, dtype=np.uint8)
    W, n = words.shape
    m = n - k
    if m % Z or k % Z:
        raise ValueError(f"n - k = {m} and k = {k} are not multiples of Z = {Z}")
    q, G = m//Z, k//Z
    use = G + extra
    if W < use + 8:
        raise ValueError(f"need about {use + 8} words, have {W}")
    P = words[:, k:]
    D = P.copy()
    D[:, 1:] ^= P[:, :-1] # d_r = p_r + p_{r-1}
    order = np.argsort(_expected_errors(rel), kind="stable") if rel is not None else np.arange(W)
    held_out = order[use + max(0, (trials - 1)*4):][:256]
    rng = np.random.default_rng(seed)
    best_A, best_h = None, np.full(q, -1.0)
    rows_r = (np.arange(q)[:, None] + q*np.arange(Z)[None, :]) # row r of class c, offset j
    for t in range(trials):
        pool = order[:use + (trials - 1)*4]
        sub = pool[:use] if t == 0 else rng.choice(pool, use, replace=False)
        M = np.concatenate([_qc_rows(words[w, :k], Z, G) for w in sub]) # (use Z, k)
        R = np.concatenate([D[w][rows_r].T for w in sub]) # (use Z, q): class c's right side at (frame, j)
        E, piv = rref(np.concatenate((M, R), axis=1), pivot_cols=k)
        if len(piv) < k:
            continue # the frames do not span: more frames, or not this k / Z
        X = E[:k, k:] # (k, q): row of pivot column = that unknown
        sol = np.zeros((k, q), np.uint8)
        sol[piv] = X
        # each class's rows on held-out frames
        h = np.zeros(q)
        for c in range(q):
            nz = np.flatnonzero(sol[:, c])
            g, t0 = nz//Z, nz % Z
            ok = []
            for w in held_out[:64]:
                u = words[w, :k].reshape(G, Z)
                lhs = np.zeros(Z, np.uint8)
                for gg, tt in zip(g, t0):
                    lhs ^= np.roll(u[gg], tt) # u[g, (j - t) mod Z] over j
                ok.append(np.mean(lhs == D[w][rows_r[c]]))
            h[c] = np.mean(ok) if ok else 0.0
        better = h > best_h
        if best_A is None:
            best_A = sol.copy()
        best_A[:, better] = sol[:, better]
        best_h[better] = h[better]
    if best_A is None:
        raise ValueError("no subset of frames spans the information bits: not an IRA code with this k and Z")
    table = [[] for _ in range(G)]
    for c in range(q):
        for v in np.flatnonzero(best_A[:, c]):
            g, j0 = divmod(int(v), Z)
            table[g].append(c + q*j0)
    table = [sorted(t) for t in table]
    from ....code.types.ira import ira_checks
    return QCStaircase(table=table, checks=ira_checks(n, k, table, Z), holds=best_h, Z=Z, n=n, k=k)

def _shifts(v, Z, G):
    """Every cyclic shift (within each block of Z) of a base row v, (Z, n)"""
    V = v.reshape(G, Z)
    return np.stack([np.roll(V, j, axis=1).reshape(-1) for j in range(Z)])

def qc_dual(words, Z, extra=8, trials=200, seed=0):
    """The circulant check blocks of a quasi-cyclic code (every column block of Z, rows and columns cycled alike: 802.11n,
    802.16e, 5G NR's lifted base graphs) from aligned clean codewords (W, n), about n/Z + `extra` of them. The circulant
    system's null space holds every check row (a base row's shifts are base rows too); its sparse vectors, found by
    information sets over it, are the blocks' rows. Returns base rows (b, n) uint8, one per circulant block, lightest
    first: row j of a block is its base row with every block of Z cycled by j"""
    words = np.asarray(words, dtype=np.uint8)
    W, n = words.shape
    if n % Z:
        raise ValueError(f"n = {n} is not a multiple of Z = {Z}")
    G = n//Z
    use = min(W, G + extra)
    M = np.concatenate([_qc_rows(words[w], Z, G) for w in range(use)])
    R, piv = rref(M)
    pset = set(piv)
    free = [c for c in range(n) if c not in pset]
    if not free:
        return np.zeros((0, n), np.uint8)
    B = np.zeros((len(free), n), np.uint8)
    for i, f in enumerate(free): # null vector: x_f = 1, the pivots from their rows
        B[i, f] = 1
        B[i, piv] = R[:, f]
    # (column (g, t) of the system is the check's position Z g + (j - t) mod Z at row j: row 0 has it at (g, -t))
    g, t = np.divmod(np.arange(n), Z)
    B0 = np.zeros_like(B)
    B0[:, Z*g + (-t) % Z] = B
    # sparse vectors of the null space: information sets (random column orders, reduced: rows one free column each)
    rng = np.random.default_rng(seed)
    found = {}
    for _ in range(trials):
        perm = rng.permutation(n)
        rows, _ = rref(B0[:, perm])
        rows_full = np.zeros_like(rows)
        rows_full[:, perm] = rows
        for r_ in rows_full:
            wgt = int(r_.sum())
            if wgt and wgt <= 2*G:
                key = min(tuple(np.flatnonzero(sh)) for sh in _shifts(r_, Z, G)) # one per shift class
                found.setdefault(key, r_)
    cand = sorted(found.values(), key=lambda v: int(v.sum()))
    out, span = [], np.zeros((0, n), np.uint8)
    target = rank(B0)
    for v in cand:
        S = np.concatenate((span, _shifts(v, Z, G)))
        if rank(S) > rank(span) if len(span) else True:
            out.append(v)
            span = S
            if rank(span) >= target:
                break
    return np.array(out, dtype=np.uint8) if out else np.zeros((0, n), np.uint8)

def qc_size(words, candidates=None, extra=8):
    """Circulant sizes Z (dividing n) at which aligned clean codewords (W, n) have checks that stay checks cycled within
    every block of Z: {Z: how many independent ones}. The code's own Z has them all (n - k); its divisors and multiples
    some, other sizes none: the largest is the code's"""
    words = np.asarray(words, dtype=np.uint8)
    W, n = words.shape
    cands = candidates or [z for z in range(8, n//4 + 1) if n % z == 0]
    out = {}
    for Z in cands:
        G = n//Z
        if W < G + extra:
            continue
        M = np.concatenate([_qc_rows(words[w], Z, G) for w in range(G + extra)])
        nul = n - rank(M)
        if nul:
            out[Z] = nul
    return out
