"""Blind LDPC and algebraic block codes, past what one search finds: staircase (IRA) codes, progressive reconstruction, roots

    H, holds = rp.analysis.ldpc.staircase(words, k)          # dual-diagonal parity (DVB-S2, 802.11n, 802.16e, 5G NR)
    H, holds, rounds = rp.analysis.ldpc.progressive(llr)     # checks found, frames cleaned with them, more found, ...
    r = rp.analysis.ldpc.roots(words)                        # a binary cyclic code's roots: BCH and its distance

staircase: an IRA code's parity part is dual-diagonal: parity bit j is parity bit j-1 plus a sum of information bits, the
row of A in H = [A | staircase]. With k information bits first, d_j = p_j + p_{j-1} is a linear function of them, the same
for every word: over k + some words, one GF(2) elimination of the information bits solves every row at once (no search
for sparse dual words: Valembois, Cluzeau, Canteaut-Chabaud would look for each). Noisy bits: the most reliable words are
used, several subsets, and each row is kept by its vote and checked on words it was not solved from.

progressive: the checks one search finds (rpps.analysis.block.sparse_checks: information sets) decode the frames (min-sum
on the partial H), the decoded frames have fewer errors, and the next search finds checks the first could not, until no
new ones come (Cluzeau 2006, block code reconstruction with iterative decoding; the ISIT 2024 progressive QC-LDPC
reconstruction reaches ~100x the bit error rate of one search).

roots: a binary cyclic code of length n | 2^m - 1 is its generator's roots: every codeword c has c(beta^j) = 0 at them
(the Vandermonde parity checks). Evaluated at every beta^j, words of the code are zero at the roots and a fraction 2^-m
elsewhere; a word with an error is not zero there either, so the zero fraction is the clean words' share, which stands far
over 2^-m at a few percent of clean words. Consecutive roots give a BCH code's designed distance.
"""
import numpy as np

from .. import _accel
from ..math.gf import GF
from ..math.gf2 import rref
from .block import sparse_checks

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

def posterior(H, llr, iters=20, alpha=0.8):
    """A-posteriori LLRs of words (B, n) through the checks H (c, n) by normalized min-sum (flooding, all checks at once):
    what the checks known so far say about each bit. The C++ kernel ldpc_flood runs the same loop (this is its reference)"""
    H = np.asarray(H, dtype=bool)
    llr = np.asarray(llr, dtype=np.float64)
    deg = H.sum(axis=1)
    dmax = int(deg.max())
    idx = np.zeros((len(H), dmax), dtype=np.int64)
    valid = np.arange(dmax)[None, :] < deg[:, None]
    for i, h in enumerate(H):
        idx[i, :deg[i]] = np.flatnonzero(h)
    B, n = llr.shape
    nat = _accel.native
    if nat is not None and hasattr(nat, "ldpc_flood"):
        return nat.ldpc_flood(np.ascontiguousarray(llr), idx, deg.astype(np.int64), int(iters), float(alpha))
    msg = np.zeros((B, len(H), dmax))
    total = llr.copy()
    flat = idx[valid]
    for _ in range(iters):
        q = total[:, idx] - msg # (B, c, dmax): variable to check, without the check's own message
        a = np.where(valid[None], np.abs(q), np.inf)
        neg = (q < 0) & valid[None]
        sgn = np.where(neg.sum(axis=2, keepdims=True) % 2 == 1, -1.0, 1.0)*np.where(neg, -1.0, 1.0)
        i1 = np.argmin(a, axis=2)[..., None]
        m1 = np.take_along_axis(a, i1, axis=2)
        a2 = a.copy()
        np.put_along_axis(a2, i1, np.inf, axis=2)
        m2 = a2.min(axis=2, keepdims=True)
        m2 = np.where(np.isinf(m2), m1, m2)
        msg = np.where(valid[None], alpha*sgn*np.where(np.arange(dmax)[None, None, :] == i1, m2, m1), 0.0)
        total = llr.copy()
        np.add.at(total.T, flat, msg[:, valid].T)
    return total

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
    found so far (see the module). kw: rpps.analysis.block.sparse_checks's. Stops when a round adds no rank.
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

def _packed(B):
    """Rows of bits (r, c) uint8 -> (r, words) uint64, column i at word i // 64, bit 63 - i % 64"""
    B = np.asarray(B, dtype=np.uint8)
    r, c = B.shape
    pad = (-c) % 64
    if pad:
        B = np.concatenate((B, np.zeros((r, pad), np.uint8)), axis=1)
    return np.ascontiguousarray(np.packbits(B, axis=1)).view(">u8").astype(np.uint64).reshape(r, -1)

def _unpacked(A, c):
    return np.unpackbits(A.astype(">u8").view(np.uint8), axis=1)[:, :c]

def _eliminate(A, ncols):
    """Reduced row echelon form in place over GF(2) of packed rows A, the first ncols columns as pivots: pivot columns.
    The C++ kernel (gf2_eliminate, Four Russians) gives the same matrix; this loop is its reference"""
    nat = _accel.native
    if nat is not None and hasattr(nat, "gf2_eliminate") and A.dtype == np.uint64 and A.flags.c_contiguous:
        return list(nat.gf2_eliminate(A, int(ncols)))
    return _eliminate_python(A, ncols)

def _eliminate_python(A, ncols):
    r = 0
    piv = []
    R = A.shape[0]
    for col in range(ncols):
        if r == R:
            break
        w, b = col >> 6, np.uint64(63 - (col & 63))
        bits = (A[r:, w] >> b) & np.uint64(1)
        nz = np.flatnonzero(bits)
        if not len(nz):
            continue
        p = r + int(nz[0])
        if p != r:
            A[[r, p]] = A[[p, r]]
        hit = np.flatnonzero((A[:, w] >> b) & np.uint64(1))
        hit = hit[hit != r]
        if len(hit):
            A[hit, w:] ^= A[r, w:]
        piv.append(col)
        r += 1
    return piv

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
        A = _packed(np.concatenate((M, R), axis=1))
        piv = _eliminate(A, k)
        if len(piv) < k:
            continue # the frames do not span: more frames, or not this k / Z
        X = _unpacked(A[:k], k + q)[:, k:] # (k, q): row of pivot column = that unknown
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
    from ..code.types.ira import ira_checks
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
    from ..math.gf2 import rank
    words = np.asarray(words, dtype=np.uint8)
    W, n = words.shape
    if n % Z:
        raise ValueError(f"n = {n} is not a multiple of Z = {Z}")
    G = n//Z
    use = min(W, G + extra)
    M = np.concatenate([_qc_rows(words[w], Z, G) for w in range(use)])
    A = _packed(M)
    piv = _eliminate(A, n)
    pset = set(piv)
    free = [c for c in range(n) if c not in pset]
    if not free:
        return np.zeros((0, n), np.uint8)
    R = _unpacked(A[:len(piv)], n)
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
        Ap = _packed(B0[:, perm])
        p2 = _eliminate(Ap, n)
        rows = _unpacked(Ap[:len(p2)], n)
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
        A = _packed(M)
        nul = n - len(_eliminate(A, n))
        if nul:
            out[Z] = nul
    return out
