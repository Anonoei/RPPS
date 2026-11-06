import numpy as np

def ira_checks(n, k, table, group=360):
    """Parity checks of a DVB-S2 style irregular repeat-accumulate LDPC code from its address table

    table: one row of parity addresses per group of `group` information bits. Information bit j of group i adds to
    parity bits (x + (j mod group) * q) mod (n-k) for every address x in the row, q = (n-k)/group, and the parity
    bits are accumulated (p[m] ^= p[m-1]). Returns [positions] per check, information bits first.
    """
    m = n - k
    q = m // group
    if q*group != m:
        raise ValueError(f"n-k={m} is not a multiple of the group size {group}")
    if len(table) * group != k:
        raise ValueError(f"{len(table)} table rows of {group} bits do not cover k={k}")
    cols = [[] for _ in range(m)]
    for i, row in enumerate(table):
        for x in row:
            addr = (np.asarray(x) + np.arange(group)*q) % m
            for j, a in enumerate(addr):
                cols[a].append(i*group + j)
    checks = []
    for c in range(m):
        v = cols[c] + [k + c]
        if c > 0:
            v.append(k + c - 1)
        checks.append(np.array(v, dtype=np.int64))
    return checks

def ira_encode(bits, n, k, table, group=360):
    """Systematic encoder for the same construction: [information | parity]"""
    m = n - k
    q = m // group
    u = np.asarray(bits, dtype=np.uint8).reshape(-1, k)
    out = np.zeros((len(u), n), dtype=np.uint8)
    out[:, :k] = u
    p = np.zeros((len(u), m), dtype=np.uint8)
    for i, row in enumerate(table):
        for x in row:
            addr = (np.asarray(x) + np.arange(group)*q) % m
            p[:, addr] ^= u[:, i*group:(i+1)*group]
    p = np.bitwise_xor.accumulate(p, axis=1)
    out[:, k:] = p
    return out.reshape(-1).astype(bool)
