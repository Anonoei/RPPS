"""Reed-Muller codes RM(r, m): the evaluations of every Boolean polynomial of degree at most r in m variables

    {"base": "block", "type": "rm", "r": 1, "m": 5}       RM(1, 5): (32, 6, 16), Mariner 9's code

n = 2^m, k = sum over i <= r of C(m, i), minimum distance 2^(m-r). Codeword position p is the point whose variable j is
bit j of p; the message is the polynomial's coefficients, monomials by degree (1, x_0 .. x_{m-1}, x_0 x_1, ...), each
degree's in lexicographic order of their variables.

Decoding: RM(1, m) by the fast Hadamard transform, maximum likelihood on LLRs (the correlation with every affine
function at once: Green's "Green machine" of Mariner 9); higher orders by Reed's majority logic on hard decisions, the
top degree first (each monomial's coefficient is the majority of the 2^(m - d) sums over the subcubes its variables
span), which corrects up to 2^(m-r-1) - 1 errors. `errors`: bits the chosen codeword differs from the hard decisions.
"""
import itertools

import numpy as np

from ...math.gf2 import fwht
from ._code import _code, register, as_llr, word_errors

def monomials(r, m):
    """The variable sets of the monomials of degree <= r, in message order"""
    return [s for d in range(r + 1) for s in itertools.combinations(range(m), d)]

@register("rm")
class rm(_code):
    soft = True

    def __init__(self, log, r, m):
        self.r, self.m = int(r), int(m)
        if not 0 <= self.r <= self.m or self.m < 1:
            raise ValueError(f"RM({r}, {m}): needs 0 <= r <= m, m >= 1")
        n = 2**self.m
        self.mono = monomials(self.r, self.m)
        pts = (np.arange(n)[:, None] >> np.arange(self.m)[None, :]) & 1 # (n, m) variable values at each position
        self.G = np.array([np.prod(pts[:, list(s)], axis=1) if s else np.ones(n, np.int64) for s in self.mono], np.uint8)
        super().__init__(log, f"rm_{self.r}_{self.m}", len(self.mono), n)

    @classmethod
    def from_config(cls, log, obj):
        return cls(log, obj["r"], obj["m"])

    def encode(self, bits):
        msg = np.asarray(bits).astype(np.int64).reshape(-1, self.num)
        return ((msg @ self.G) % 2).reshape(-1).astype(bool)

    def decode(self, bits):
        x = np.asarray(bits)
        llr = as_llr(x[:len(x)//self.den*self.den]).reshape(-1, self.den)
        msg = self._fht(llr) if self.r == 1 else self._majority((llr < 0).astype(np.uint8))
        self.errors = word_errors(llr < 0, (msg.astype(np.int64) @ self.G) % 2)
        return msg.reshape(-1).astype(bool)

    def _fht(self, llr):
        """RM(1, m): the affine function nearest each word. The transform of the LLRs at u is the correlation with the
        linear function u.x; its largest magnitude picks u, its sign the constant term"""
        out = np.zeros((len(llr), self.num), np.uint8)
        for w, row in enumerate(llr):
            F = fwht(row)
            u = int(np.argmax(np.abs(F)))
            out[w, 0] = F[u] < 0
            out[w, 1:] = (u >> np.arange(self.m)) & 1
        return out

    def _majority(self, y):
        """Reed's majority logic, degree by degree from the top (y (W, n) hard bits)"""
        W, n = y.shape
        y = y.copy()
        msg = np.zeros((W, self.num), np.uint8)
        p = np.arange(n)
        for d in range(self.r, -1, -1):
            idx = [i for i, s in enumerate(self.mono) if len(s) == d]
            for i in idx:
                mask = sum(1 << v for v in self.mono[i])
                order = np.argsort(p & ~mask, kind="stable") # positions grouped by the variables outside the monomial
                sums = np.bitwise_xor.reduce(y[:, order].reshape(W, n >> d, 1 << d), axis=2) # (W, 2^(m-d)) votes
                msg[:, i] = 2*sums.sum(axis=1) > (n >> d) # (a tie decides 0)
            y ^= ((msg[:, idx].astype(np.int64) @ self.G[idx].astype(np.int64)) % 2).astype(np.uint8)
        return msg
