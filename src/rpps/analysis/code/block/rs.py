"""Blind Reed-Solomon identification: length, field, generator and alignment of an unknown RS code over GF(256)

An RS codeword c(x) = sum c_i x^(n-1-i) (first byte the highest power) is a multiple of its generator
g(x) = prod (x - alpha^(prim*(fcr+i))), i < n-k, so c(beta) = 0 at those roots, whatever the shortening (a shortened word is
a full one with leading zeros). The Galois-field Fourier transform of an aligned word, its values at all 255 nonzero
beta = alpha^j, is zero at the roots in every word; elsewhere, and for any word that is not aligned, a value is zero 1/256
of the time. So the roots show up as the j where nearly every word is zero, and they give n-k, fcr and prim.

Every byte phase is evaluated at once, with a sliding window: E_(s+1)(beta) = beta*E_s(beta) + b[s+n] + b[s]*beta^n. This is
done for all 16 primitive field polynomials, 8 bit offsets, both bit orders within a byte (MSB or LSB first) and both
polarities (inverted bits add the constant 0xFF*sum beta^i to every value) together.

What is needed: the length n, from `lengths` (hints, the frame period of the bits, common lengths) or tried
exhaustively; at least `words` + 2 codewords of bits with few errors. Symbol interleaving (depth I: codeword bytes every
I-th byte) is undone by `depths`.
"""
import math

import numpy as np

from ...result import Hypothesis, Guess
from ...framing.bits import frame_length
from ....math.prob import deviance_z
from ....math.gf import GF

PRIMITIVE = (0x11D, 0x12B, 0x12D, 0x14D, 0x15F, 0x163, 0x165, 0x169, 0x171, 0x187, 0x18D, 0x1A9, 0x1C3, 0x1CF, 0x1E7, 0x1F5)
COMMON = (255, 204, 208, 207, 219, 225, 146, 126, 194, 160, 120) # DVB, ATSC, IESS-308/309/310, DSS, CCSDS (255), ...
_REV = np.array([int(f"{i:08b}"[::-1], 2) for i in range(256)], dtype=np.uint8)

def _tables(polys):
    """EXP (P, 510) and LOG (P, 256) of each field"""
    fields = [GF(8, poly) for poly in polys]
    return np.stack([g.exp[:510] for g in fields]), np.stack([g.log for g in fields])

def _variants(bits, depths):
    """Byte streams for every bit offset, bit order and interleaving depth: (labels, (V, L) bytes)"""
    b = np.asarray(bits, dtype=np.uint8)
    labels, rows = [], []
    for off in range(8):
        B = np.packbits(b[off:][:(len(b) - off)//8*8])
        for lsb in (False, True):
            BB = _REV[B] if lsb else B
            for I in depths:
                for c in range(I):
                    labels.append({"bit_offset": off, "lsb_first": lsb, "depth": I, "branch": c})
                    rows.append(BB[c::I])
    L = min(len(r) for r in rows)
    return labels, np.stack([r[:L] for r in rows]).astype(np.int64)

def _units():
    return [u for u in range(1, 255) if math.gcd(u, 255) == 1]

def fit_generator(roots):
    """(fcr, prim) with roots = {prim*(fcr+i) mod 255, i < len(roots)}, prim 1 if it can be, or None"""
    r = len(roots)
    R = set(int(x) for x in roots)
    for prim in _units():
        inv = pow(prim, -1, 255)
        idx = sorted((e*inv) % 255 for e in R)
        for start in idx:
            if {(start + i) % 255 for i in range(r)} == set(idx):
                return start, prim
    return None

def _mul_tables(EXP, LOG, power):
    """M[P, j, x] = x * alpha^(j*power) in each field, for every j < 255 and byte x"""
    P = EXP.shape[0]
    p = np.arange(P)[:, None, None]
    by = np.arange(256)[None, None, :]
    e = (LOG[p, by] + (np.arange(255)*power)[None, :, None]) % 255
    return np.where(by == 0, 0, EXP[p, e]).astype(np.uint8)

def _count_zeros(V, n, EXP, LOG, words):
    """Zero counts of the sliding transform: (n phases, V variants, P fields, 255 j) for normal and inverted polarity"""
    P = EXP.shape[0]
    Vn, L = V.shape
    M1 = _mul_tables(EXP, LOG, 1).reshape(-1) # x * beta, flat: [(P*255 + j)*256 + x]
    MN = _mul_tables(EXP, LOG, n).reshape(-1) # x * beta^n, for the byte leaving the window
    base = ((np.arange(P)[:, None]*255 + np.arange(255)[None, :])*256).astype(np.int32)[None] # (1, P, 255)
    Q = np.zeros((P, 255), dtype=np.int32) # 0xFF * sum_i beta^i (Horner): what inverted bits add
    for _ in range(n):
        Q = np.take(M1, base[0] + Q) ^ 0xFF
    Q = Q.astype(np.uint8)[None]
    from .... import _accel
    if _accel.native is not None: # The loop below in C++ (rpps._native.rs_count_zeros), the same counts
        return _accel.native.rs_count_zeros(np.ascontiguousarray(V, dtype=np.uint8), int(n), np.ascontiguousarray(M1),
                                            np.ascontiguousarray(MN), np.ascontiguousarray(Q[0]), int(words))
    V = V.astype(np.int32)
    E = np.zeros((Vn, P, 255), dtype=np.int32)
    z0 = np.zeros((n, Vn, P, 255), dtype=np.int16)
    z1 = np.zeros((n, Vn, P, 255), dtype=np.int16)
    for t in range(min(L, n*(words + 1))):
        E = np.take(M1, base + E).astype(np.int32)
        E ^= V[:, t][:, None, None]
        if t >= n:
            E ^= np.take(MN, base + V[:, t - n][:, None, None])
        if t >= n - 1:
            s = (t - n + 1) % n
            z0[s] += E == 0
            z1[s] += E == Q
    return z0, z1

def _verify(stream, n, k, poly, fcr, prim, phase, invert, words=64):
    """Words at this alignment whose syndromes are all zero (one repeated value doesn't count), out of how many"""
    from ....logger import Logger
    from ....code.types.rs import rs
    code = rs(Logger().Child("blind-rs", 0), k, n, 8, poly, fcr, prim)
    seg = stream[phase:]
    W = min(words, len(seg)//n)
    if W < 2:
        return 0, 0
    w = seg[:W*n].reshape(W, n) ^ (0xFF if invert else 0)
    flat = (w == w[:, :1]).all(axis=1)
    ok = ~code.syndromes(w).any(axis=1) & ~flat
    return int(ok.sum()), int(W)

def lengths_of(bits, hints=()):
    """Lengths to try: the hints, then the byte frame periods of the bits, then COMMON"""
    out = list(hints)
    for nb, _ in frame_length(np.asarray(bits)[:2**17]):
        if nb % 8 == 0:
            out.append(nb//8)
    return list(dict.fromkeys(n for n in out + list(COMMON) if 8 <= n <= 255))

def search(bits, lengths=None, hints=(), words=8, depths=(1,), polys=PRIMITIVE, exhaustive=False):
    """Find an RS code over GF(256) in hard bits. Returns (hypotheses best first, trials, Guess or None)

    lengths: codeword lengths in bytes to try (default: `lengths_of`); exhaustive: every length 16..255 (minutes)
    words: codewords per phase the roots are counted on (a root is zero in at least 3/4 of them)
    Each hypothesis: params n, k, poly, fcr, prim, roots, phase (byte where a codeword starts), bit_offset, lsb_first,
    depth, branch, polarity, words_ok (fraction of further words whose syndromes are all zero)."""
    lens = list(range(16, 256)) if exhaustive else (list(lengths) if lengths else lengths_of(bits, hints))
    labels, V = _variants(bits, depths)
    EXP, LOG = _tables(polys)
    hyps, trials, seen = [], 0, set()
    for n in lens:
        if V.shape[1] < n*(words + 2):
            continue
        z0, z1 = _count_zeros(V, n, EXP, LOG, words)
        need = max(3, math.ceil(0.75*words))
        trials += n*V.shape[0]*len(polys)*2
        for invert, z in ((False, z0), (True, z1)):
            hit = z >= need
            r = hit.sum(axis=3)
            cand = (r >= 2) & (r <= n//2)
            # A code fits at one codeword phase. One that fits at several is not one: words of one repeated value (idle,
            # a dead carrier) are zero at the n-th roots of unity wherever the window is
            cand &= cand.sum(axis=0, keepdims=True) <= 1
            for phase, v, p in zip(*np.nonzero(cand)):
                roots = np.nonzero(hit[phase, v, p])[0]
                gen = fit_generator(roots)
                if gen is None:
                    continue
                fcr, prim = gen
                k = n - len(roots)
                lab = labels[v]
                key = (n, k, polys[p], fcr, prim, lab["depth"], invert)
                if key in seen: # The same code at another bit offset / branch of the same bits
                    continue
                ok, W = _verify(V[v], n, k, polys[p], fcr, prim, int(phase), invert)
                if W == 0 or ok < 2:
                    continue
                seen.add(key)
                zz = float(deviance_z(ok, W, 256.0**-len(roots)))
                hyps.append(Hypothesis("rs-blind", f"RS({n},{k})", zz,
                    {"n": n, "k": k, "poly": polys[p], "fcr": int(fcr), "prim": int(prim), "roots": roots.tolist(),
                     "phase": int(phase), "polarity": int(invert), "words_ok": ok/W, **lab}))
    hyps.sort(key=lambda h: (-h.params["words_ok"], -h.z))
    guess = None
    if hyps:
        p = hyps[0].params
        guess = Guess("rs", {"base": "block", "type": "rs", "comment": f"found by rpps.analysis.code.block.rs.search: {hyps[0].name}",
                             "n": p["n"], "k": p["k"], "m": 8, "poly": p["poly"], "fcr": p["fcr"], "prim": p["prim"]},
                      {k: p[k] for k in ("phase", "bit_offset", "lsb_first", "depth", "polarity", "words_ok")})
    return hyps, trials, guess
