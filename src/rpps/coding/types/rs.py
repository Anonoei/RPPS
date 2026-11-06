import numpy as np

from ._code import _code, register
from ...math import cyclic
from ...math.gf import GF

@register("rs")
class rs(_code):
    """Reed-Solomon code over GF(2^m), shortened to (n, k): symbols are bytes for m=8

    poly: field polynomial (0x11D). fcr/prim: first consecutive root and primitive element power of the generator
    (DVB: fcr=0, prim=1, 16 roots). Bits are grouped MSB first into m-bit symbols.
    encode/decode take bits (or a byte array with symbols=True) of whole codewords.
    After a decode, `errors` is the number of corrected symbols per codeword (-1: uncorrectable).
    """
    @classmethod
    def from_config(cls, log, obj):
        return cls(log, obj["k"], obj["n"], obj.get("m", 8), obj.get("poly", 0x11D), obj.get("fcr", 0), obj.get("prim", 1))

    def __init__(self, log, num: int, den: int, m=8, poly=0x11D, fcr=0, prim=1):
        super().__init__(log, None, num, den)
        self.gf = GF(m, poly)
        self.m, self.fcr, self.prim = m, fcr, prim
        self.nroots = den - num
        self.t = self.nroots // 2
        self.errors = None
        # Generator polynomial, highest degree first
        g = np.array([1], dtype=np.int64)
        for i in range(self.nroots):
            root = self.gf.pow_alpha((fcr + i)*prim)
            g = self.gf.polymul(g, np.array([1, root]))
        self.gen = g

    def to_symbols(self, bits):
        bits = np.asarray(bits).astype(np.uint8)
        bits = bits[:len(bits)//self.m*self.m]
        return (bits.reshape(-1, self.m) * (1 << np.arange(self.m-1, -1, -1))).sum(axis=1)

    def to_bits(self, symbols):
        s = np.asarray(symbols, dtype=np.int64)
        return ((s[:, None] >> np.arange(self.m-1, -1, -1)) & 1).astype(bool).reshape(-1)

    def syndromes(self, words):
        """Syndromes (W, nroots) of codewords (W, n): S_i = r(alpha^((fcr+i)*prim))"""
        words = np.asarray(words)
        e = np.arange(words.shape[1] - 1, -1, -1) # Power of x at each position
        S = np.zeros((len(words), self.nroots), dtype=np.int64)
        for i in range(self.nroots):
            S[:, i] = self.gf.evaluate(words, e*(self.fcr + i)*self.prim)
        return S

    def encode(self, bits, symbols=False):
        msg = np.asarray(bits) if symbols else self.to_symbols(bits)
        msg = msg[:len(msg)//self.num*self.num].reshape(-1, self.num).astype(np.int64)
        out = np.zeros((len(msg), self.den), dtype=np.int64)
        out[:, :self.num] = msg
        rem = msg.copy()
        rem = np.concatenate((rem, np.zeros((len(msg), self.nroots), dtype=np.int64)), axis=1)
        for i in range(self.num):
            coef = rem[:, i].copy()
            nz = coef != 0
            for j in range(1, len(self.gen)):
                rem[nz, i+j] ^= self.gf.mul(coef[nz], self.gen[j])
        out[:, self.num:] = rem[:, self.num:]
        return out.reshape(-1) if symbols else self.to_bits(out.reshape(-1))

    def decode(self, bits, symbols=False):
        cw = (np.asarray(bits) if symbols else self.to_symbols(bits)).astype(np.int64)
        cw = cw[:len(cw)//self.den*self.den].reshape(-1, self.den).copy()
        S = self.syndromes(cw)
        self.errors = np.zeros(len(cw), dtype=int)
        for w in np.nonzero(S.any(axis=1))[0]:
            fixed = self._correct(cw[w], S[w])
            self.errors[w] = -1 if fixed is None else fixed
        out = cw[:, :self.num].reshape(-1)
        return out if symbols else self.to_bits(out)

    def _correct(self, word, S):
        """Fix one word in place. Returns the number of corrected symbols, None if it can't be corrected"""
        found = cyclic.correct(self.gf, S, self.den, self.nroots, self.t, self.fcr, self.prim)
        if found is None:
            return None
        positions, values = found
        word[positions] ^= values
        if self.syndromes(word[None, :]).any():
            return None
        return len(positions)
