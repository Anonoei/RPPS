import numpy as np

from ._code import register
from .linear import linear
from ...math.gf2 import parity_to_gh

@register("cyclic")
class cyclic(linear):
    """Binary cyclic code from its generator polynomial, systematic with the message first: Golay, Hamming, BCH...

    poly: g(x) as an integer, bit i the coefficient of x^i (Golay (23,12): 0xC75), degree n - k, dividing x^n + 1.
    extend: append an overall parity bit (Golay (24,12)). shorten: drop that many leading message bits (sent as 0).
    t: errors the decoder should correct (default: from the minimum distance).
    A codeword is the message followed by the remainder of m(x) x^(n-k) mod g(x), highest power first.
    """
    @classmethod
    def from_config(cls, log, obj):
        poly = obj["poly"]
        return cls(log, obj["n"], obj["k"], int(poly, 0) if isinstance(poly, str) else poly, obj.get("extend", False),
                   obj.get("shorten", 0), obj.get("t"))

    def __init__(self, log, n, k, poly, extend=False, shorten=0, t=None):
        r = n - k
        if poly.bit_length() - 1 != r:
            raise ValueError(f"cyclic ({n},{k}): generator {poly:#x} has degree {poly.bit_length()-1}, need n-k={r}")
        if _mod2(1 << n | 1, poly) != 0:
            raise ValueError(f"cyclic ({n},{k}): {poly:#x} does not divide x^{n}+1")
        if not 0 <= shorten < k:
            raise ValueError(f"cyclic ({n},{k}): can't shorten by {shorten}")
        p = np.array([[(_mod2(1 << (r + k - 1 - i), poly) >> (r - 1 - j)) & 1 for j in range(r)] for i in range(k)], dtype=np.uint8)
        if extend: # Overall parity: of the message bit and its parity row
            p = np.concatenate((p, ((1 + p.sum(axis=1)) % 2)[:, None]), axis=1)
        p = p[shorten:]
        kk = k - shorten
        G, H = parity_to_gh(kk + p.shape[1], kk, p)
        super().__init__(log, G, H)
        self.poly, self.extend, self.shorten, self.t = poly, extend, shorten, t

def _mod2(a, g):
    """a(x) mod g(x) over GF(2), polynomials as integers"""
    dg = g.bit_length() - 1
    while a and a.bit_length() - 1 >= dg:
        a ^= g << (a.bit_length() - 1 - dg)
    return a
