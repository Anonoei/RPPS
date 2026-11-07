"""Algebraic decoding of cyclic codes (Reed-Solomon, BCH) from their syndromes

Berlekamp-Massey finds the error locator polynomial, a Chien search finds its roots (the error positions), and
Forney finds the error values. A binary code needs no values: every error flips a bit.

Syndromes are S_i = r(alpha^((fcr + i)*prim)), i = 0..nroots-1, of a received word r of length N. Position 0 is the
first symbol of the word (the highest power of x). Polynomials are arrays of field elements, lowest degree first
unless noted.
"""
import numpy as np

def berlekamp_massey(gf, S, nroots):
    """Error locator sigma (nroots+1 coefficients, lowest degree first) and its degree L from the syndromes"""
    sigma = np.zeros(nroots + 1, dtype=np.int64)
    sigma[0] = 1
    prev = sigma.copy()
    L, m, b = 0, 1, 1
    for n in range(nroots):
        d = int(S[n])
        for i in range(1, L + 1):
            d ^= int(gf.mul(sigma[i], S[n - i]))
        if d == 0:
            m += 1
            continue
        T = sigma.copy()
        shifted = np.zeros(nroots + 1, dtype=np.int64)
        shifted[m:] = gf.mul(int(gf.mul(d, gf.inv(b))), prev[:nroots + 1 - m])
        sigma = sigma ^ shifted
        if 2*L <= n:
            L, prev, b, m = n + 1 - L, T, d, 1
        else:
            m += 1
    return sigma, L

def chien(gf, sigma, L, N, prim=1):
    """Positions of a word of length N whose locator root is sigma(alpha^(-(N-1-pos)*prim)) = 0"""
    e = np.arange(N - 1, -1, -1) * prim
    val = np.zeros(N, dtype=np.int64)
    for i in range(L + 1):
        if sigma[i]:
            val ^= gf.exp[(gf.log[sigma[i]] - e*i) % (gf.size - 1)]
    return np.nonzero(val == 0)[0]

def forney(gf, S, sigma, L, positions, N, nroots, fcr=0, prim=1):
    """Error values at `positions`. None where the derivative of sigma vanishes (the word is not correctable)"""
    omega = np.zeros(nroots, dtype=np.int64) # S(x) sigma(x) mod x^nroots
    for i in range(nroots):
        for j in range(min(i + 1, L + 1)):
            omega[i] ^= int(gf.mul(sigma[j], S[i - j]))
    values = []
    for pos in positions:
        x = int(gf.pow_alpha((N - 1 - pos)*prim))
        xinv = int(gf.inv(x))
        num = 0
        for k, c in enumerate(omega):
            num ^= int(gf.mul(c, gf.pow(xinv, k)))
        den = 0
        for i in range(1, L + 1, 2): # Formal derivative: the odd terms
            den ^= int(gf.mul(sigma[i], gf.pow(xinv, i - 1)))
        if den == 0:
            return None
        values.append(int(gf.mul(gf.mul(num, gf.inv(den)), gf.pow(x, 1 - fcr))))
    return np.array(values, dtype=np.int64)

def correct(gf, S, N, nroots, t, fcr=0, prim=1, binary=False):
    """Find the errors of one word from its syndromes. Returns (positions, values) or None if uncorrectable

    t: errors the code corrects. binary: values are all 1 and not computed (BCH)
    """
    sigma, L = berlekamp_massey(gf, S, nroots)
    if L > t:
        return None
    positions = chien(gf, sigma, L, N, prim)
    if len(positions) != L:
        return None
    if binary:
        return positions, np.ones(L, dtype=np.int64)
    values = forney(gf, S, sigma, L, positions, N, nroots, fcr, prim)
    return None if values is None else (positions, values)
