import numpy as np

from ... import _accel
from ._code import _code, register
from ...math import cyclic
from ...math.gf import GF

@register("bch")
class bchgf(_code):
    """Binary BCH code over GF(2^m), shortened to (n, k), as in DVB-S2 (m=16, or 14 for short frames)

    poly: field polynomial (0x1002D = 1+x^2+x^3+x^5+x^16). t: correctable bit errors. The generator is the product of
    the distinct minimal polynomials of alpha^1, alpha^3, ..., alpha^(2t-1), so n-k = m*t.
    encode/decode take whole codewords as bits. After a decode `errors` is the corrected bit count per word (-1: failed).
    """
    @classmethod
    def from_config(cls, log, obj):
        return cls(log, obj["n"], obj["k"], obj.get("m", 16), int(obj.get("poly", "0x1002D"), 0), obj["t"])

    def __init__(self, log, n, k, m=16, poly=0x1002D, t=12):
        super().__init__(log, None, k, n)
        self.m, self.t, self.poly = m, t, poly
        self.gf = GF(m, poly)
        self.size, self.exp, self.log = self.gf.size, self.gf.exp, self.gf.log
        self.gen = self._generator()
        if len(self.gen) - 1 != n - k:
            raise ValueError(f"BCH generator has degree {len(self.gen)-1}, but n-k={n-k}")
        self.errors = None

    def mul(self, a, b):
        return self.gf.mul(a, b)

    def _generator(self):
        """Binary generator polynomial, highest degree first"""
        seen, g = set(), np.array([1], dtype=np.int64)
        for i in range(1, 2*self.t, 2):
            if i in seen:
                continue
            conj, e = [], i
            while e not in conj:
                conj.append(e)
                e = (e*2) % (self.size - 1)
            seen.update(conj)
            mp = np.array([1], dtype=np.int64) # Product of (x + alpha^e) over the conjugates
            for e in conj:
                mp = self.gf.polymul(mp, np.array([1, self.exp[e]]))
            g = self.gf.polymul(g, mp)
        return g.astype(np.uint8)

    def encode(self, bits):
        u = np.asarray(bits, dtype=np.uint8).reshape(-1, self.num)
        r = self.den - self.num
        out = np.zeros((len(u), self.den), dtype=np.uint8)
        out[:, :self.num] = u
        rem = np.concatenate((u, np.zeros((len(u), r), dtype=np.uint8)), axis=1)
        for i in range(self.num): # Polynomial division, one message bit at a time
            nz = rem[:, i] == 1
            rem[nz, i:i+r+1] ^= self.gen
        out[:, self.num:] = rem[:, self.num:]
        return out.reshape(-1).astype(bool)

    def syndromes(self, words):
        """S_j = r(alpha^j), j=1..2t, for words (W, n) of bits"""
        words = np.asarray(words)
        if _accel.native is not None:
            return _accel.native.gf_syndromes(np.ascontiguousarray(words, dtype=np.int64), self.gf.exp, self.gf.log, 2*self.t, 1, 1)
        e = np.arange(self.den - 1, -1, -1) # Power of x of each bit
        S = np.zeros((len(words), 2*self.t), dtype=np.int64)
        for j in range(1, 2*self.t + 1):
            S[:, j-1] = self.gf.evaluate(words, e*j)
        return S

    def decode(self, bits):
        cw = np.asarray(bits, dtype=np.uint8)
        cw = cw[:len(cw)//self.den*self.den].reshape(-1, self.den).copy()
        if _accel.native is not None: # The codewords on several threads
            out, errors = _accel.native.cyclic_decode(cw.astype(np.int64), np.zeros((0, 0), np.uint8), self.gf.exp, self.gf.log,
                                                      2*self.t, self.t, 1, 1, True, False)
            self.errors = errors.astype(int)
            return out[:, :self.num].reshape(-1).astype(bool)
        S = self.syndromes(cw)
        self.errors = np.zeros(len(cw), dtype=int)
        for w in np.nonzero(S.any(axis=1))[0]:
            pos = self._locate(S[w])
            if pos is None:
                self.errors[w] = -1
            else:
                cw[w, pos] ^= 1
                self.errors[w] = len(pos)
        return cw[:, :self.num].reshape(-1).astype(bool)

    def _locate(self, S):
        """Error positions of one word from its syndromes, or None if uncorrectable"""
        found = cyclic.correct(self.gf, S, self.den, 2*self.t, self.t, fcr=1, binary=True)
        return None if found is None else found[0]
