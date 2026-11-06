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

def nullspace(M):
    """Basis of {h : M h = 0} over GF(2). M is (R, n) uint8. Returns (d, n) uint8"""
    A = np.array(M, dtype=np.uint8)
    R, n = A.shape
    piv = []
    r = 0
    for c in range(n):
        if r == R:
            break
        rows = np.nonzero(A[r:, c])[0]
        if len(rows) == 0:
            continue
        p = rows[0] + r
        if p != r:
            A[[r, p]] = A[[p, r]]
        others = np.nonzero(A[:, c])[0]
        others = others[others != r]
        A[others] ^= A[r]
        piv.append(c)
        r += 1
    free = [c for c in range(n) if c not in set(piv)]
    basis = np.zeros((len(free), n), dtype=np.uint8)
    for i, f in enumerate(free):
        basis[i, f] = 1
        for row, c in enumerate(piv):
            basis[i, c] = A[row, f]
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
