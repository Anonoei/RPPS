"""Linear algebra over GF(2)"""
import numpy as np

def parity_to_gh(n: int, k: int, p: np.ndarray):
    """Generator and parity-check matrices (G, H) of a systematic linear code from its parity block

    p is (k, n-k), the parity part of G = [I | p]; H = [p^T | I]. Returns uint8 arrays (k, n) and (n-k, n).
    """
    d = n - k
    p = np.asarray(p)
    if p.shape != (k, d):
        raise IndexError(f"p must be shape ({k},{d})!")
    G = np.concatenate((np.eye(k, dtype=np.uint8), p.astype(np.uint8)), axis=1)
    H = np.concatenate((p.T.astype(np.uint8), np.eye(d, dtype=np.uint8)), axis=1)
    return G, H

def gh_to_parity(n: int, k: int, G=None, H=None):
    """Parity block p (k, n-k) of a systematic code from G = [I | p], or from H = [p^T | I]"""
    if G is not None:
        return np.asarray(G)[:k, k:]
    if H is not None:
        return np.asarray(H)[:n-k, :k].T
    raise ValueError("Need G or H")

def to_checks(h):
    """Dense parity-check matrix -> list of variable index arrays (one per check)"""
    return [np.nonzero(row)[0] for row in np.asarray(h, dtype=bool)]

def _pack(M):
    """Rows of M (R, n) of 0/1 packed into uint64 words, bit c of a row at byte c//8, bit 7 - c%8: (A64, A8 view of it)"""
    M = np.asarray(M, dtype=np.uint8)
    R, n = M.shape
    words = max(1, (n + 63)//64)
    A8 = np.zeros((R, words*8), dtype=np.uint8)
    A8[:, :(n + 7)//8] = np.packbits(M, axis=1)
    return A8.view(np.uint64), A8

def pack_rows(B):
    """Rows of bits (r, c) -> (r, words) uint64, column i at word i // 64, bit 63 - i % 64: what `eliminate_packed` takes"""
    B = np.asarray(B, dtype=np.uint8)
    r, c = B.shape
    pad = (-c) % 64
    if pad or not c:
        B = np.concatenate((B, np.zeros((r, pad or 64), np.uint8)), axis=1)
    return np.ascontiguousarray(np.packbits(B, axis=1)).view(">u8").astype(np.uint64).reshape(r, -1)

def unpack_rows(A, c):
    """`pack_rows` undone: (r, c) uint8"""
    return np.unpackbits(np.ascontiguousarray(A).astype(">u8").view(np.uint8), axis=1)[:, :c]

def eliminate_packed(A, ncols):
    """Reduced row echelon form in place over GF(2) of packed rows A (`pack_rows`), the first ncols columns as pivots: the
    pivot columns. The C++ kernel (gf2_eliminate, Four Russians) gives the same matrix bit for bit; the loop below is its
    reference"""
    from .. import _accel
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
        if len(hit): # Row r is zero left of column col (earlier columns are pivots, cleared, or zero below their turn)
            A[hit, w:] ^= A[r, w:]
        piv.append(col)
        r += 1
    return piv

def rref(M, pivot_cols=None):
    """Reduced row echelon form of M over GF(2), bit-packed: (rows (r, n) uint8, pivot columns). r is the rank.
    pivot_cols: pivot only within the first that many columns (an augmented system [A | B]: the rows give A^-1 B)"""
    M = np.asarray(M, dtype=np.uint8)
    R, n = M.shape
    A = pack_rows(M)
    piv = eliminate_packed(A, n if pivot_cols is None else min(pivot_cols, n))
    return unpack_rows(A[:len(piv)], n), piv

def rank(M):
    """Rank of M over GF(2) (bit-packed elimination, rows below the pivot only)"""
    M = np.asarray(M, dtype=np.uint8)
    R, n = M.shape
    A64, A8 = _pack(M)
    r = 0
    for c in range(n):
        if r == R:
            break
        byte, bit = c >> 3, 7 - (c & 7)
        nz = np.flatnonzero((A8[r:, byte] >> bit) & 1)
        if not len(nz):
            continue
        p = nz[0] + r
        if p != r:
            A64[[r, p]] = A64[[p, r]]
        below = r + 1 + np.flatnonzero((A8[r + 1:, byte] >> bit) & 1)
        if len(below):
            A64[below] ^= A64[r]
        r += 1
    return r

def nullspace(M):
    """Basis of {h : M h = 0} over GF(2). M is (R, n) uint8. Returns (d, n) uint8"""
    M = np.asarray(M, dtype=np.uint8)
    n = M.shape[1]
    A, piv = rref(M)
    pset = set(piv)
    free = [c for c in range(n) if c not in pset]
    basis = np.zeros((len(free), n), dtype=np.uint8)
    if not free:
        return basis
    basis[np.arange(len(free)), free] = 1
    if piv:
        basis[:, piv] = A[:, free].T # Each pivot variable is the sum of the free ones its row holds
    return basis

def systematic(h):
    """GF(2) elimination of dense H. Returns (G, info) with H @ G.T = 0 and the message in codeword columns `info`"""
    H = np.array(h, dtype=bool)
    M, N = H.shape
    perm = np.arange(N)
    r = 0
    for c in range(N):
        if r == M:
            break
        piv = np.nonzero(H[r:, c])[0]
        if len(piv) == 0:
            continue
        piv = piv[0] + r
        H[[r, piv]] = H[[piv, r]]
        others = np.nonzero(H[:, c])[0]
        others = others[others != r]
        H[others] ^= H[r]
        H[:, [r, c]] = H[:, [c, r]] # Move the pivot column to position r
        perm[[r, c]] = perm[[c, r]]
        r += 1
    rank = r
    A = H[:rank, rank:]
    K = N - rank
    Gp = np.concatenate((A.T, np.eye(K, dtype=bool)), axis=1) # Codeword in permuted order: [parity | message]
    G = np.zeros((K, N), dtype=bool)
    G[:, perm] = Gp
    return G, perm[rank:]

# ---- Polynomials over GF(2) as Python ints (bit i: the coefficient of x^i)

def pmod(a, b):
    """a(x) mod b(x) over GF(2), polynomials as Python ints (bit i: the coefficient of x^i)"""
    db = b.bit_length()
    while a.bit_length() >= db:
        a ^= b << (a.bit_length() - db)
    return a

def pdivmod(a, b):
    """(quotient, remainder) of a(x)/b(x) over GF(2)"""
    q, db = 0, b.bit_length()
    while a.bit_length() >= db:
        sh = a.bit_length() - db
        q |= 1 << sh
        a ^= b << sh
    return q, a

def pmul(a, b):
    """a(x) b(x) over GF(2)"""
    out = 0
    while b:
        if b & 1:
            out ^= a
        a <<= 1
        b >>= 1
    return out

def pgcd(*polys):
    """Greatest common divisor over GF(2) of polynomials"""
    g = 0
    for p in polys:
        a, b = g, int(p)
        while b:
            a, b = b, pmod(a, b)
        g = a
    return g

def porder(g, limit=1 << 17):
    """Smallest N with x^N = 1 mod g(x) (the period of the recurrence g gives, the length of the cyclic code it
    generates), or None up to `limit` (g of degree >= 1 with g(0) = 1)"""
    d = g.bit_length() - 1
    if d < 1 or not g & 1:
        return None
    x = 1
    for i in range(1, limit + 1):
        x <<= 1
        if (x >> d) & 1:
            x ^= g
        if x == 1:
            return i
    return None

def berlekamp_massey(bits):
    """The shortest LFSR that makes a bit sequence: (C, L), C its connection polynomial (bit j the coefficient of D^j,
    bit 0 set: s[n] = XOR of s[n-j] over the j >= 1 where C has bit j) and L its length (the linear complexity).
    `lfsr_taps(C, L)` gives the j's"""
    C, B = 1, 1
    L, m = 0, 1
    window = 0 # bit j: s[n - j]
    for n, b in enumerate(bits):
        window = (window << 1) | (int(b) & 1)
        d = (C & window).bit_count() & 1 # s[n] + sum C_j s[n-j]
        if d == 0:
            m += 1
        elif 2*L <= n:
            T = C
            C ^= B << m
            L, B, m = n + 1 - L, T, 1
        else:
            C ^= B << m
            m += 1
    return C, L

def fwht(a):
    """Walsh-Hadamard transform (unnormalized) of a length-2^B array, a copy: the correlation of a distribution over B-bit
    words with every parity check (what the blind TCM, recurrence and RSC searches rank checks by)"""
    a = np.array(a, dtype=np.float64)
    n = len(a)
    h = 1
    while h < n:
        a = a.reshape(-1, 2, h)
        a = np.stack((a[:, 0] + a[:, 1], a[:, 0] - a[:, 1]), axis=1).reshape(n)
        h *= 2
    return a

def lfsr_taps(C, L):
    """The delays j (largest first) of a connection polynomial: s[n] = XOR of s[n-j]"""
    return tuple(j for j in range(L, 0, -1) if C >> j & 1)

def rank_ints(vectors):
    """Rank over GF(2) of vectors given as Python ints"""
    basis = []
    for v in vectors:
        for b in basis:
            v = min(v, v ^ b)
        if v:
            basis.append(v)
    return len(basis)
