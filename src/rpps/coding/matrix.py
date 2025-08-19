import numpy as np

def make_id(size: int):
    idm = np.zeros((size, size))
    idm[np.arange(size),np.arange(size)] = 1
    return idm

def linear(n: int, k: int, p: np.ndarray):
    # n: length
    # k: data_bits
    # p: parity matrix
    d = n-k # redundant

    if not (p.shape == (k,d) or p.shape == (d,k)):
        raise IndexError(f"p must be shape ({k},{d}) or ({d},{k})")

    # print(f"Linear Matrix [{n}, {k}, {d}]")

    if not p.shape == (k,d):
        p = p.T
    g_id = make_id(k)
    g_matrix = np.zeros((k,n))
    for i in range(k):
        g_matrix[i,0:k] = g_id[i]
        g_matrix[i,k:] = p[i]
    # print(f"generator:\n{g_matrix.astype(int)}")

    if not p.shape == (d,k):
        p = p.T
    c_id = make_id(d)
    h_matrix = np.zeros((d,n))
    for i in range(d):
        h_matrix[i,0:k] = p[i]
        h_matrix[i,k:] = c_id[i]
    # print(f"parity check:\n{h_matrix.astype(int)}")
    return g_matrix, h_matrix
