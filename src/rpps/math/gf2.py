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

def rref(M):
    """Reduced row echelon form of M over GF(2), bit-packed: (rows (r, n) uint8, pivot columns). r is the rank"""
    M = np.asarray(M, dtype=np.uint8)
    R, n = M.shape
    A64, A8 = _pack(M)
    piv = []
    r = 0
    for c in range(n):
        if r == R:
            break
        byte, bit = c >> 3, 7 - (c & 7)
        col = (A8[:, byte] >> bit) & 1
        nz = np.flatnonzero(col[r:])
        if not len(nz):
            continue
        p = nz[0] + r
        if p != r:
            A64[[r, p]] = A64[[p, r]]
            col[[r, p]] = col[[p, r]]
        others = np.flatnonzero(col)
        others = others[others != r]
        if len(others):
            A64[others] ^= A64[r]
        piv.append(c)
        r += 1
    return np.unpackbits(A8[:r], axis=1)[:, :n], piv

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
