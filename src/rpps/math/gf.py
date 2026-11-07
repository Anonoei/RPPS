"""Finite field GF(2^m) arithmetic"""
import numpy as np

class GF:
    """GF(2^m) with log tables

    poly: the field polynomial, with the x^m term included (0x11D for m=8). Elements are ints below 2^m, and every
    operation takes scalars or numpy arrays. Polynomials are arrays of elements, highest degree first.
    """
    def __init__(self, m=8, poly=0x11D):
        self.m = m
        self.poly = poly
        self.size = 2**m
        self.exp = np.zeros(2*self.size, dtype=np.int64)
        self.log = np.zeros(self.size, dtype=np.int64)
        x = 1
        for i in range(self.size-1):
            self.exp[i] = x
            self.log[x] = i
            x <<= 1
            if x & self.size:
                x ^= poly
        self.exp[self.size-1:2*self.size-2] = self.exp[:self.size-1]

    def mul(self, a, b):
        a, b = np.asarray(a), np.asarray(b)
        r = self.exp[(self.log[a] + self.log[b]) % (self.size-1)]
        return np.where((a == 0) | (b == 0), 0, r)

    def inv(self, a):
        return self.exp[(self.size-1 - self.log[a]) % (self.size-1)]

    def pow_alpha(self, e):
        """alpha^e, for the primitive element alpha"""
        return self.exp[e % (self.size-1)]

    def pow(self, a, e):
        """a^e for a single element"""
        if e == 0:
            return 1
        if a == 0:
            return 0
        return int(self.exp[(int(self.log[a])*e) % (self.size-1)])

    def polymul(self, a, b):
        """Product of two polynomials"""
        a, b = np.asarray(a), np.asarray(b)
        out = np.zeros(len(a) + len(b) - 1, dtype=np.int64)
        for i, x in enumerate(a):
            out[i:i+len(b)] ^= self.mul(x, b)
        return out

    def evaluate(self, words, powers):
        """Sum over positions of words[:, k] * alpha^powers[k]: the polynomial values at a power of alpha, for every word

        words: (W, N) field elements. powers: (N,) exponents of alpha, one per position
        """
        words = np.asarray(words)
        terms = self.exp[(self.log[words] + np.asarray(powers)[None, :]) % (self.size - 1)]
        return np.bitwise_xor.reduce(np.where(words == 0, 0, terms), axis=1)
