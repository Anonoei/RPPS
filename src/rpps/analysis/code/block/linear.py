"""Blind binary block codes: length, alignment, parity checks, and which kind of code it is (cyclic / BCH, product, LDPC)

Rows of L bits from a stream of codewords of length n: when L is a multiple of n (and for most alignments when L = n), every
row holds whole codewords at the same place, so the rows satisfy the code's parity checks and the matrix of rows is short of
full rank. Random bits, or rows of any other length, are not (with `EXTRA` more rows than columns, a random matrix is short
2^-EXTRA of the time). That finds the length; the alignment is where rows of n bits are shortest of rank; the checks are the
null space of the rows (`recover_dual`, which tolerates some words with errors).

Then what kind of code:
 cyclic (BCH, cyclic Hamming and Golay, shortened ones too): every codeword is a multiple of the generator g(x), so g is the
    greatest common divisor of codewords. Its full length is the order of x modulo g; a narrow-sense BCH code is the one whose
    generator is the product of the minimal polynomials of alpha, alpha^3, ..., alpha^(2t-1) in some GF(2^m).
 product (turbo product codes): rows are codewords of one code and, stacked n2 at a time, columns of another.
 LDPC: the checks found have low weight (`sparse_checks`, which takes soft values to pick the cleanest words).

The rank methods need about n + EXTRA rows of n bits with few errors: lengths up to roughly sqrt(len(bits)), at a bit error
rate well below 1/n^2 (after a decoder, or a strong signal). What they find is a Guess: decoded once saved.
"""
from collections import Counter

import numpy as np

from ....math.gf2 import rank, nullspace, pmod, pgcd, porder, berlekamp_massey, lfsr_taps
from ....math.prob import z_critical
from ...result import Hypothesis, Guess

EXTRA = 48 # Rows beyond the row length

def deficiency(bits, L, offset=0, segments=3, extra=EXTRA):
    """L minus the rank of rows of L bits from `offset`, the largest over `segments` stretches of the bits (an error in a row
    adds to the rank, so the cleanest stretch counts). 0 for random bits; the number of independent checks for a code. None
    when there are fewer than L + extra rows"""
    b = np.asarray(bits, dtype=np.uint8)[offset:]
    W = L + extra
    total = len(b)//L
    if total < W:
        return None
    starts = sorted(set(np.linspace(0, total - W, segments).astype(int))) if segments > 1 else [0]
    return max(L - rank(b[s*L:(s + W)*L].reshape(W, L)) for s in starts)

def alignment(bits, n, extra=EXTRA):
    """(offset, deficiency at each offset): where rows of n bits are shortest of rank"""
    d = np.array([deficiency(bits, n, o, segments=2, extra=extra) or 0 for o in range(n)])
    return int(np.argmax(d)), d

def find_length(bits, min_len=4, max_len=None, extra=EXTRA):
    """Codeword length of a binary block code: (n, offset, {length: deficiency}) or (None, None, deficiencies)

    The rank deficiency shows at multiples of n (and at n itself for most alignments): n is the smallest divisor of the
    first deficient length at which some alignment is deficient. Structure at most lengths (constant bits, a repetition)
    is not a block code."""
    b = np.asarray(bits, dtype=np.uint8)
    max_len = max_len or int(np.sqrt(len(b)/1.2)) - extra
    defs = {}
    for L in range(min_len, max_len + 1):
        d = deficiency(b, L, extra=extra)
        if d is None:
            break
        defs[L] = d
    hits = [L for L, d in defs.items() if d > 0]
    if not hits or len(hits) > 0.5*len(defs):
        return None, None, defs
    first = hits[0]
    for n in [d for d in range(min_len, first + 1) if first % d == 0]:
        off, prof = alignment(b, n, extra)
        # Most deficient at one alignment, above the others (a sparse code keeps some checks at any alignment; constant
        # bits are as deficient everywhere)
        if prof[off] > 0 and prof[off] > np.median(prof):
            return _longer(b, n, off, max_len, extra) + (defs,)
    return None, None, defs

def _longer(b, n, off, max_len, extra):
    """A code of length n may be part of a longer one (LDPC: a band of checks on n consecutive bits): rows of m*n bits short
    of rank by more than m times rows of n are. Returns (length, offset), the longest such, aligned"""
    d1 = deficiency(b, n, off, extra=extra) or 0
    for m in range(2, max_len//n + 1):
        L = m*n
        best = max(range(0, L, n), key=lambda o: deficiency(b, L, off + o, segments=2, extra=extra) or 0)
        d = deficiency(b, L, off + best, extra=extra)
        if d is None:
            break
        if d > m*d1:
            return _longer(b, L, off + best, max_len, extra)
    return n, off

# --- Polynomials over GF(2) as Python ints (bit i: the coefficient of x^i) -----------------------------------------------

def to_poly(row):
    """Bits, first the highest power, as a polynomial int"""
    row = np.asarray(row, dtype=np.uint8)
    pad = (-len(row)) % 8
    return int.from_bytes(np.packbits(row).tobytes(), "big") >> pad

def cyclic_generator(words, r):
    """g(x) of a cyclic code from aligned codewords (W, n): the commonest gcd of two words that has degree r (= n - k). With
    errors in some words their gcds differ; codewords' share g. Returns (g, fraction of words g divides) or (None, 0)"""
    polys = [to_poly(w) for w in words]
    cnt = Counter()
    for a, b in zip(polys[0::2], polys[1::2]):
        if a and b:
            g = pgcd(a, b)
            if g.bit_length() - 1 == r:
                cnt[g] += 1
    if not cnt:
        return None, 0.0
    g = cnt.most_common(1)[0][0]
    return g, float(np.mean([pmod(p, g) == 0 for p in polys]))

def primitive_polys(m):
    """Primitive polynomials of degree m (x is a generator of GF(2^m))"""
    N = (1 << m) - 1
    return [p for p in range((1 << m) + 1, 1 << (m + 1), 2) if porder(p, N) == N]

def bch_of(g, n_sent):
    """The narrow-sense BCH definition with generator g, if it is one: {"type": "bch", ...} or None"""
    from ....logger import Logger
    from ....code.types.bchgf import bchgf
    N = porder(g)
    r = g.bit_length() - 1
    if N is None:
        return None
    m = next((m for m in range(2, 17) if ((1 << m) - 1) % N == 0), None)
    if m is None or N != (1 << m) - 1 or r % m:
        return None
    for p in primitive_polys(m) if m <= 16 else []:
        if pmod(g, p):
            continue
        try:
            gen = bchgf(Logger().Child("blind-bch", 0), n_sent, n_sent - r, m, p, r//m).gen
        except ValueError:
            continue
        if to_poly(gen) == g:
            return {"base": "block", "type": "bch", "n": int(n_sent), "k": int(n_sent - r), "m": m, "poly": hex(p), "t": r//m}
    return None

# --- Product codes ----------------------------------------------------------------------------------------------------

def product(bits, n1, offset, max_n2=128, extra=24):
    """Rows of n1 bits stacked n2 at a time, whose columns are codewords too: (n2, row phase, deficiency) or None"""
    b = np.asarray(bits, dtype=np.uint8)[offset:]
    rows = b[:len(b)//n1*n1].reshape(-1, n1)
    best = None
    for n2 in range(3, max_n2 + 1):
        d = []
        for ph in range(n2):
            B = (len(rows) - ph)//n2
            if B < n2 + extra:
                break
            # One column from each block: words from many blocks (a few blocks can agree with a wrong n2 by chance)
            cols = rows[ph:ph + B*n2].reshape(B, n2, n1).transpose(0, 2, 1)[np.arange(B), np.arange(B) % n1]
            d.append(n2 - rank(cols[:n2 + extra]))
        if len(d) < n2:
            break
        d = np.array(d)
        ph = int(np.argmax(d))
        if d[ph] > 0 and (d == d[ph]).sum() == 1 and (best is None or d[ph] > best[2]):
            best = (n2, ph, int(d[ph]))
            if (d > 0).sum() == 1: # One phase only: the column length (a multiple would come later, and weaker per row)
                return best
    return best

# --- LDPC: low-weight checks ------------------------------------------------------------------------------------------

def sparse_checks(words, rel=None, w_max=None, trials=150, verify=400, seed=0, k=None, patience=20):
    """Low-weight parity checks of aligned words (W, n): (H (c, n) uint8, fraction of words each holds on)

    Each trial puts the columns in a random order and takes the null space of up to k + 16 words: its basis vectors are each
    one free position plus pivots, and a sparse check comes out as one of them whenever all but one of its positions are
    pivots. Those of weight <= w_max that hold on far more than half of `verify` words are kept. rel: reliabilities
    (|LLR|) of the bits: the words with the fewest expected errors (sum of 1/(1 + e^|LLR|)) are the ones used. k: the code's
    dimension if known: k + 16 words span it, and the fewer words the likelier they are all clean."""
    words = np.asarray(words, dtype=np.uint8)
    W, n = words.shape
    w_max = w_max or max(6, min(40, n//4))
    R = min(W, (k if k is not None else n) + 16)
    # Checks are verified on words they were not computed from (a null space vector holds on those by construction)
    order = (np.argsort((1.0/(1.0 + np.exp(np.minimum(np.asarray(rel, dtype=float), 50.0)))).sum(axis=1), kind="stable")
             if rel is not None else np.arange(W))
    npool = min(4*R, max(R, W - 64))
    pool = words[order[:npool]]
    rest = order[npool:]
    if len(rest) < 32:
        return np.zeros((0, n), np.uint8), np.zeros(0)
    test = words[np.random.default_rng(seed + 1).choice(rest, min(len(rest), verify), replace=False)].astype(np.int32)
    crit = z_critical(max(1, trials*n))
    # (trials: the most; it stops once `patience` trials in a row find nothing new)
    rng = np.random.default_rng(seed)
    found = {}
    quiet = 0
    for t in range(trials):
        if quiet >= patience: # Nothing new for a while: what can be found has been
            break
        before = len(found)
        # The cleanest words first; then random subsets of the reliable ones, some smaller than k (likelier all clean: their
        # null space holds junk too, which fails verification, and the sparse checks still come out as basis vectors)
        if t == 0:
            sub = pool[:R]
        else:
            size = int(rng.integers(max(4, R//2), R + 1))
            sub = pool[rng.choice(len(pool), min(size, len(pool)), replace=False)]
        perm = rng.permutation(n)
        N = nullspace(sub[:, perm])
        if not len(N):
            quiet += 1
            continue
        h = np.zeros_like(N)
        h[:, perm] = N
        h = h[h.sum(axis=1) <= w_max]
        if not len(h):
            quiet += 1
            continue
        sat = 1.0 - ((test @ h.T.astype(np.int32)) % 2).mean(axis=0)
        z = (sat - 0.5)*2*np.sqrt(len(test))
        for v, s in zip(h[z > crit], sat[z > crit]):
            found.setdefault(v.tobytes(), (v, float(s)))
        quiet = 0 if len(found) > before else quiet + 1
    if not found:
        return np.zeros((0, n), np.uint8), np.zeros(0)
    H = np.array([v for v, s in found.values()])
    return H, np.array([s for v, s in found.values()])

def lightest(H):
    """The sparsest description of the checks H span: a basis taken lightest first, then every check no heavier than its
    heaviest (redundant low-weight checks help a decoder: Gallager's H has more rows than its rank)"""
    H = np.asarray(H, dtype=np.uint8)
    if not len(H):
        return H
    H = H[np.argsort(H.sum(axis=1), kind="stable")]
    basis, r = [], 0
    for row in H:
        if rank(np.array(basis + [row])) > r:
            basis.append(row)
            r += 1
    w = max(int(b.sum()) for b in basis)
    return H[H.sum(axis=1) <= w]

def _product(b, n1, k1, H1, n2, start):
    """(Hypothesis, Guess) of the product of rows (n1, k1; checks H1) with columns of n2 rows, blocks from bit `start`"""
    rows = b[start:]
    rows = rows[:len(rows)//(n1*n2)*(n1*n2)].reshape(-1, n2, n1)
    cols = rows.transpose(0, 2, 1).reshape(-1)
    rc = recover_dual(cols, n2, 0, trials=20)
    if not rc["ok"]:
        return None
    k2 = rc["k"]
    part = lambda nn, kk, HH: {"type": "ldpc", "n": int(nn), "k": int(kk), "checks": _checks(HH)}
    ev = {"n1": int(n1), "k1": int(k1), "n2": int(n2), "k2": int(k2), "offset": int(start)}
    return (Hypothesis("block-blind", f"product ({n1},{k1}) x ({n2},{k2})", float(rows.shape[0]), {**ev, "kind": "product"}),
            Guess("block", {"base": "block", "type": "product", "rows": part(n1, k1, H1), "cols": part(n2, k2, rc["H"]),
                            "comment": "found by rpps.analysis.code.block.linear.solve_block (where the message bits sit is not checked)"}, ev))

def _product_of(b, n, k, offset):
    """A code of length n that is a product: rows of a divisor n1 are codewords, columns of n/n1 too, and k = k1*k2"""
    for n1 in [d for d in range(3, n//3 + 1) if n % d == 0]:
        if not (deficiency(b, n1, offset, extra=EXTRA) or 0):
            continue
        r1 = recover_dual(b, n1, offset, trials=20)
        if not r1["ok"]:
            continue
        found = _product(b, n1, r1["k"], r1["H"], n//n1, offset)
        if found is not None and found[0].params["k1"]*found[0].params["k2"] == k:
            return found
    return None

# --- All of it --------------------------------------------------------------------------------------------------------

def trivial_checks(words, nk, hold=0.97):
    """When checks of weight 1 and 2 (bits that hold one value, pairs that are equal or opposite, in `hold` of the words)
    account for all nk independent checks: (fixed positions, equal pairs); else (None, None)"""
    W = np.asarray(words[:400], dtype=np.int8)
    n = W.shape[1]
    m = W.mean(axis=0)
    fixed = [int(j) for j in np.nonzero((m <= 1 - hold) | (m >= hold))[0]]
    rows = []
    for j in fixed:
        r = np.zeros(n, dtype=np.uint8)
        r[j] = 1
        rows.append(r)
    rest = [j for j in range(n) if j not in set(fixed)]
    same = []
    if 1 < len(rest) <= 512:
        X = 2*W[:, rest].astype(np.float32) - 1
        C = np.abs(X.T @ X)/len(W)
        for a, c in zip(*np.nonzero(np.triu(C >= 2*hold - 1, 1))):
            r = np.zeros(n, dtype=np.uint8)
            r[rest[a]] = r[rest[c]] = 1
            rows.append(r)
            same.append((int(rest[a]), int(rest[c])))
    if not rows or rank(np.array(rows)) < nk:
        return None, None
    return fixed, same

LOW_RATE = 4 # Codes of rate under 1/LOW_RATE get the looser fill tests

def _majority(words):
    return (np.asarray(words[:400]).mean(axis=0) >= 0.5).astype(np.uint8)

def recurrence(bits, n, windows=4, hold=0.9):
    """(degree, taps, fraction holding) when the bits follow one short linear recurrence (shorter than n: what makes every
    row of n bits short of rank, at every alignment), else None. Berlekamp-Massey on a few clean-looking stretches of
    3n bits; the shortest recurrence found must hold over the whole stream"""
    from ...sequence.scrambler import recurrence_holds
    b = np.asarray(bits, dtype=np.uint8)
    span = 3*n
    if len(b) < 2*span:
        return None
    best = None
    for s in np.linspace(0, len(b) - span, windows).astype(int):
        C, L = berlekamp_massey(b[s:s + span])
        taps = lfsr_taps(C, L)
        if 0 < L < n and taps and (best is None or L < best[0]):
            best = (L, tuple(taps))
    if best is None:
        return None
    h = float(np.mean(recurrence_holds(b, best[1])))
    return (best[0], best[1], h) if h >= hold else None

def _checks(H):
    return [list(map(int, np.nonzero(row)[0])) for row in np.asarray(H, dtype=bool)]

def solve_block(bits, n=None, offset=None, llr=None, max_len=None, max_n2=128):
    """Find a binary block code in bits: its length, alignment, checks, and kind. Returns [(Hypothesis, Guess or None)]

    n / offset: when known (a frame length, a hint); else found by rank (lengths up to max_len, by default what the bits
    allow). llr: soft values of the bits (positive = 0), to pick the cleanest words for the sparse checks."""
    b = np.asarray(bits, dtype=np.uint8)
    if n is None:
        n, offset, _ = find_length(b, max_len=max_len)
        if n is None:
            return []
    elif offset is None:
        offset, _ = alignment(b, n)
    r = recover_dual(b, n, offset, trials=20)
    if not r["ok"]:
        return []
    H, k = r["H"], r["k"]
    if k < 1: # Every word the same: a pattern repeating every n bits (idle fill, a carrier), not a code
        return []
    W = (len(b) - offset)//n
    words = b[offset:offset + W*n].reshape(W, n)
    sat = 1.0 - ((words[:400].astype(np.int32) @ H.T.astype(np.int32)) % 2).mean()
    base = {"n": int(n), "k": int(k), "offset": int(offset), "checks_hold": float(sat)}
    z = float((sat - 0.5)*2*np.sqrt(min(W, 400)*len(H)))
    fixed, same = trivial_checks(words, n - k)
    if fixed is None and k <= n//LOW_RATE: # A code this low a rate is rare; fill with a few bit errors is not
        fixed, same = trivial_checks(words, n - k, hold=0.9)
    if fixed is not None: # Bits that never change, or copy one another, every n bits: a frame's structure, not a code
        return [(Hypothesis("frame-blind", f"period {n} bits: {len(fixed)} bits fixed, {len(same)} pairs equal (framing or "
                            "fill, not an error-correcting code)", z, {**base, "kind": "frame", "fixed": fixed,
                                                                       "same": same}), None)]
    rec = recurrence(b, n)
    if rec is not None: # Every bit a sum of the few before it: rows of any length past that are short of rank
        L, taps, holds = rec
        return [(Hypothesis("frame-blind", f"a recurrence of degree {L} ({holds*100:.0f}% of bits): a PRBS, idle pattern or "
                            "test sequence, not an error-correcting code", z, {**base, "kind": "recurrence", "degree": L,
                                                                                 "taps": list(taps), "holds": holds}), None)]
    if k <= n//LOW_RATE:
        dist = (words != _majority(words)).mean(axis=1)
        if np.median(dist) < 0.1: # Most words the same pattern: fill, a carrier, a locked rate too high
            return [(Hypothesis("frame-blind", f"period {n} bits: most words within {np.median(dist)*100:.0f}% of one "
                                "pattern (fill, not an error-correcting code)", z, {**base, "kind": "fill"}), None)]

    prod = _product_of(b, n, k, offset)
    if prod is not None:
        return [prod]

    p = product(b, n, offset, max_n2) if n*max_n2*(max_n2 + 24) <= len(b) else None
    if p is not None: # Rows of n are a code, and so are columns of n2 of them: a product code longer than the rank could see
        n2, ph, _ = p
        found = _product(b, n, k, H, n2, offset + ph*n)
        if found is not None:
            return [found]

    g, frac = cyclic_generator(words[:200], n - k)
    if g is not None and frac > 0.5:
        N = porder(g)
        if N is not None and N < n: # g divides x^N + 1 for every multiple of its order: the smallest one that holds n bits
            N = -(-n//N)*N
        bch = bch_of(g, n)
        if bch is not None:
            return [(Hypothesis("block-blind", f"BCH({n},{k}) t={bch['t']} GF(2^{bch['m']})", z, {**base, "kind": "bch", "g": hex(g)}),
                     Guess("block", {**bch, "comment": "found by rpps.analysis.code.block.linear.solve_block"}, {**base, "words_ok": frac}))]
        hyp = Hypothesis("block-blind", f"cyclic ({n},{k}) g={g:#x}", z, {**base, "kind": "cyclic", "g": hex(g), "full_length": N})
        guess = None
        if N is not None and N <= 4096: # (the cyclic definition builds dense matrices of the full length)
            guess = Guess("block", {"base": "block", "type": "cyclic", "n": int(N), "k": int(N - (n - k)), "poly": hex(g),
                                    "shorten": int(N - n), "comment": "found by rpps.analysis.code.block.linear.solve_block"},
                          {**base, "words_ok": frac})
        return [(hyp, guess)]


    rel = None
    if llr is not None:
        rel = np.abs(np.asarray(llr, dtype=float)[offset:offset + W*n]).reshape(W, n)
    S, s_sat = sparse_checks(words, rel, k=k)
    S = lightest(S)
    if len(S) and rank(S) == n - k: # The whole dual from low-weight checks: an LDPC code (or any with a sparse dual)
        name = "LDPC" if n >= 64 else "linear, sparse checks"
        return [(Hypothesis("block-blind", f"{name} ({n},{k}) checks of weight <= {int(S.sum(axis=1).max())}", z,
                            {**base, "kind": "ldpc" if n >= 64 else "sparse", "checks": len(S)}),
                 Guess("block", {"base": "block", "type": "ldpc", "n": int(n), "k": int(k), "checks": _checks(S),
                                 "comment": "found by rpps.analysis.code.block.linear.solve_block"}, base))]
    return [(Hypothesis("block-blind", f"linear ({n},{k})", z, {**base, "kind": "linear"}),
             Guess("block", {"base": "block", "type": "ldpc", "n": int(n), "k": int(k), "checks": _checks(H),
                             "comment": "found by rpps.analysis.code.block.linear.solve_block (dense checks)"}, base))]

def recover_dual(bits, n, offset=0, trials=40, seed=0, z_min=8.0):
    """Recover parity checks of an unknown linear block code of length n from (noisy) hard bits

    Words that are codewords satisfy every parity check h. For a random set of words, the null space of
    their matrix contains the checks that no error among them broke, and anything that is a check satisfies
    the other words too: so null vectors are verified against all words, and those that stand out
    from 50% are kept. Needs about n words and a word error rate that leaves some subsets of words clean.

    Returns dict(H, k, words, needed, ok). H is a basis of the checks (dense, not sparse).
    """
    rng = np.random.default_rng(seed)
    bits = np.asarray(bits, dtype=np.uint8)
    m = (len(bits) - offset)//n
    words = bits[offset:offset+m*n].reshape(m, n)
    out = {"H": np.zeros((0, n), dtype=np.uint8), "k": n, "words": m, "needed": 2*n, "ok": False}
    if m < n:
        return out
    keep = np.zeros((0, n), dtype=np.uint8)
    rank, stale = 0, 0
    for _ in range(trials):
        best_rank = rank
        for s in (n//4, n//2, (3*n)//4, n-1):
            idx = rng.choice(m, size=min(s, m), replace=False)
            N = nullspace(words[idx])
            if len(N) == 0:
                continue
            sat = 1.0 - ((words.astype(np.uint8) @ N.T.astype(np.uint8)) % 2).mean(axis=0)
            z = (sat - 0.5)*2*np.sqrt(m)
            good = N[z > z_min]
            if len(good):
                cat = np.concatenate((keep, good))
                B = nullspace(nullspace(cat)) # Row space of cat: its dual's dual
                keep = B
                rank = len(keep)
            best_rank = max(best_rank, rank)
        stale = stale + 1 if rank == best_rank and best_rank == out["H"].shape[0] else 0
        out["H"] = keep
        if stale >= 3:
            break
    out["H"] = keep
    out["k"] = n - len(keep)
    out["ok"] = len(keep) > 0
    out["needed"] = max(2*n, 2*out["k"] + 64)
    return out
