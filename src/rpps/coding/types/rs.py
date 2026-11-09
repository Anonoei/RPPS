import numpy as np

from ... import _accel
from ._code import _code, register
from ...math import cyclic
from ...math.gf import GF

@register("rs")
class rs(_code):
    """Reed-Solomon code over GF(2^m), shortened to (n, k): symbols are bytes for m=8

    poly: field polynomial (0x11D). fcr/prim: first consecutive root and primitive element power of the generator
    (DVB: fcr=0, prim=1, 16 roots; P25: fcr=1). Bits are grouped MSB first into m-bit symbols.
    puncture: positions of the n-symbol codeword that are not sent (negative counts from the end, so [-4, -3, -2, -1]
    drops the last four parity symbols). They are decoded as erasures; a word has n - len(puncture) symbols on the wire.

    encode/decode take bits (or symbols with symbols=True) of whole codewords as sent.
    decode(..., erasures=...) also takes symbols known to be unreliable: a (words, sent symbols) bool array, or one
    list of positions per word. e erasures and r errors are corrected while 2r + e <= n - k.
    Soft input (LLRs, positive = 0): a word that fails errors-only decoding is tried again with its least reliable
    symbols erased (2, 4, ... up to n - k), the generalized minimum distance idea.
    After a decode, `errors` is the number of corrected symbols per codeword (-1: uncorrectable).
    """
    soft = True

    @classmethod
    def from_config(cls, log, obj):
        return cls(log, obj["k"], obj["n"], obj.get("m", 8), _int(obj.get("poly", 0x11D)), obj.get("fcr", 0), obj.get("prim", 1),
                   obj.get("puncture"))

    def __init__(self, log, num: int, den: int, m=8, poly=0x11D, fcr=0, prim=1, puncture=None):
        n = den
        if n > 2**m - 1:
            raise ValueError(f"RS({n}, {num}) is longer than GF(2^{m}) allows ({2**m - 1})")
        punct = sorted({p % n for p in (puncture or ())})
        super().__init__(log, None, num, n - len(punct))
        self.n = n # Codeword length before puncturing; `den` is what is sent
        self.puncture = np.array(punct, dtype=np.int64)
        self.sent = np.setdiff1d(np.arange(n), self.puncture)
        self.gf = GF(m, poly)
        self.m, self.fcr, self.prim = m, fcr, prim
        self.nroots = n - num
        if len(punct) > self.nroots:
            raise ValueError(f"RS({n}, {num}): {len(punct)} punctured symbols, more than the {self.nroots} parity symbols")
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
        """Syndromes (W, nroots) of full-length codewords (W, n): S_i = r(alpha^((fcr+i)*prim))"""
        words = np.asarray(words)
        if _accel.native is not None:
            return _accel.native.gf_syndromes(np.ascontiguousarray(words, dtype=np.int64), self.gf.exp, self.gf.log,
                                              self.nroots, self.fcr, self.prim)
        e = np.arange(words.shape[1] - 1, -1, -1) # Power of x at each position
        S = np.zeros((len(words), self.nroots), dtype=np.int64)
        for i in range(self.nroots):
            S[:, i] = self.gf.evaluate(words, e*(self.fcr + i)*self.prim)
        return S

    def encode(self, bits, symbols=False):
        msg = np.asarray(bits) if symbols else self.to_symbols(bits)
        msg = msg[:len(msg)//self.num*self.num].reshape(-1, self.num).astype(np.int64)
        out = np.zeros((len(msg), self.n), dtype=np.int64)
        out[:, :self.num] = msg
        rem = msg.copy()
        rem = np.concatenate((rem, np.zeros((len(msg), self.nroots), dtype=np.int64)), axis=1)
        for i in range(self.num):
            coef = rem[:, i].copy()
            nz = coef != 0
            for j in range(1, len(self.gen)):
                rem[nz, i+j] ^= self.gf.mul(coef[nz], self.gen[j])
        out[:, self.num:] = rem[:, self.num:]
        out = out[:, self.sent].reshape(-1)
        return out if symbols else self.to_bits(out)

    def _received(self, bits, symbols, soft):
        """Symbols (W, den) and, for soft input, each symbol's reliability (the smallest |LLR| of its bits)"""
        x = np.asarray(bits)
        if soft is None:
            soft = x.dtype.kind == "f" and not symbols
        if not soft:
            cw = (x if symbols else self.to_symbols(x)).astype(np.int64)
            return cw[:len(cw)//self.den*self.den].reshape(-1, self.den), None
        llr = x.astype(np.float64)
        per = self.den*self.m
        llr = llr[:len(llr)//per*per]
        cw = self.to_symbols(llr < 0).astype(np.int64).reshape(-1, self.den)
        rel = np.abs(llr).reshape(-1, self.den, self.m).min(axis=2)
        return cw, rel

    def _erasure_lists(self, erasures, W):
        """Per word: sent-symbol positions given as erased"""
        if erasures is None:
            return [()]*W
        if isinstance(erasures, np.ndarray) and erasures.dtype == bool:
            e = erasures.reshape(W, self.den)
            return [np.nonzero(row)[0] for row in e]
        if len(erasures) != W:
            raise ValueError(f"erasures: {len(erasures)} lists for {W} codewords")
        return list(erasures)

    def decode(self, bits, symbols=False, erasures=None, soft=None):
        rx, rel = self._received(bits, symbols, soft)
        W = len(rx)
        cw = np.zeros((W, self.n), dtype=np.int64)
        cw[:, self.sent] = rx
        given = self._erasure_lists(erasures, W)
        if _accel.native is not None:
            return self._decode_native(cw, given, rel, symbols)
        S = self.syndromes(cw)
        self.errors = np.zeros(W, dtype=int)
        for w in range(W):
            if not S[w].any():
                continue
            base = set(int(p) for p in self.sent[np.asarray(given[w], dtype=np.int64)]) | set(int(p) for p in self.puncture)
            fixed = self._correct(cw[w], S[w], base)
            if fixed is None and rel is not None:
                order = self.sent[np.argsort(rel[w], kind="stable")] # Least reliable first
                order = [int(p) for p in order if int(p) not in base]
                for extra in range(2, self.nroots - len(base) + 1, 2):
                    fixed = self._correct(cw[w], S[w], base | set(order[:extra]))
                    if fixed is not None:
                        break
            self.errors[w] = -1 if fixed is None else fixed
        out = cw[:, :self.num].reshape(-1)
        return out if symbols else self.to_bits(out)

    def _decode_native(self, cw, given, rel, symbols):
        """`decode` in C++ (rpps._native.cyclic_decode, the codewords on several threads), the soft retries included"""
        nat, gf = _accel.native, self.gf
        W = len(cw)
        erased = np.zeros((W, self.n), dtype=np.uint8)
        erased[:, self.puncture] = 1
        for w in range(W):
            if len(given[w]):
                erased[w, self.sent[np.asarray(given[w], dtype=np.int64)]] = 1
        args = (gf.exp, gf.log, self.nroots, self.t, self.fcr, self.prim, False, True)
        out, errors = nat.cyclic_decode(cw, erased if erased.any() else np.zeros((0, 0), np.uint8), *args)
        if rel is not None:
            for w in np.nonzero(errors < 0)[0]:
                base = erased[w].astype(bool)
                order = [int(p) for p in self.sent[np.argsort(rel[w], kind="stable")] if not base[p]] # Least reliable first
                for extra in range(2, self.nroots - int(base.sum()) + 1, 2):
                    mask = base.copy()
                    mask[order[:extra]] = True
                    one, err = nat.cyclic_decode(cw[w:w+1], mask[None].astype(np.uint8), *args)
                    if err[0] >= 0:
                        out[w], errors[w] = one[0], err[0]
                        break
        self.errors = errors.astype(int)
        res = out[:, :self.num].reshape(-1)
        return res if symbols else self.to_bits(res)

    def _correct(self, word, S, erasures=()):
        """Fix one full-length word in place. Returns the number of changed symbols, None if it can't be corrected"""
        found = cyclic.correct(self.gf, S, self.n, self.nroots, self.t, self.fcr, self.prim, erasures=erasures)
        if found is None:
            return None
        positions, values = found
        fixed = word.copy()
        fixed[positions] ^= values
        if self.syndromes(fixed[None, :]).any():
            return None
        word[:] = fixed
        return int(np.count_nonzero(values))

def _int(v):
    return int(v, 0) if isinstance(v, str) else v
