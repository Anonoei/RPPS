import numpy as np

def make_id(size: int):
    idm = np.zeros((size, size))
    idm[np.arange(size),np.arange(size)] = 1
    return idm

def linear(n: int, k: int, p: np.ndarray):
    """Create linear block code G and H matrices

    Args:
        n (int): block length
        k (int): message length
        p (int): parity matrix
          p must be shaped for the G matrix

    Returns:
        G: Generator matrix
        H: Parity-check matrix
    """
    d = n-k # parity bits

    if not p.shape == (k,d):
        raise IndexError(f"p must be shape ({k},{d})!")
    G_id = make_id(k)
    G = np.zeros((k,n))
    for i in range(k):
        G[i,0:k] = G_id[i]
        G[i,k:] = p[i]

    if not p.shape == (d,k):
        p = p.T
    H_id = make_id(d)
    H = np.zeros((d,n))
    for i in range(d):
        H[i,0:k] = p[i]
        H[i,k:] = H_id[i]
    return G, H

def linear_p(n, k, G=None, H=None):
    """Create parity matrix from G or H matrices

    Args:
        n (int): block length
        k (int): message length
        G: generator matrix
        H: parity-check matrix

    Returns:
        p: parity-check matrix
    """
    d = n-k # parity bits

    if not G is None:
        p = np.zeros((k,d))
        for i in range(k):
            p[i] = G[i,k:]
    elif not H is None:
        p = np.zeros((d,k))
        for i in range(d):
            p[i] = H[i,0:k]
    return p
