import numpy as np

from ._code import _code, register, as_llr
from .linear import linear
from ...math.gf2 import rank

MAX_K = 16 # Message bits of a code decoded from its whole codebook (2^k words)

def codebook(bits, k):
    """Every codeword (2^k, n) of codeword-bit expressions over m0..m{k-1} (and the bits before, c0..), message i's bits
    MSB first"""
    from ... import _config
    if not 0 < k <= MAX_K:
        raise ValueError(f"expr code: k={k}, need 1..{MAX_K} (its codebook has 2^k words)")
    msgs = ((np.arange(1 << k)[:, None] >> np.arange(k - 1, -1, -1)) & 1).astype(np.int64)
    values = {f"m{i}": msgs[:, i] for i in range(k)}
    out = np.zeros((1 << k, len(bits)), dtype=np.uint8)
    for j, e in enumerate(bits):
        v = _config.evaluate(str(e), values, f"expr code bit {j}") if isinstance(e, str) else e
        out[:, j] = np.broadcast_to(np.asarray(v, dtype=np.int64) & 1, (1 << k,))
        values[f"c{j}"] = out[:, j].astype(np.int64)
    return out

def affine(book, k):
    """(G, offset) when every codeword is offset + m G over GF(2), else None"""
    off = book[0]
    units = 1 << np.arange(k - 1, -1, -1) # message of unit vector e_i: bit i set (MSB first)
    G = book[units] ^ off
    msgs = ((np.arange(1 << k)[:, None] >> np.arange(k - 1, -1, -1)) & 1).astype(np.uint8)
    if not np.array_equal(((msgs @ G) % 2).astype(np.uint8) ^ off, book):
        return None
    return G, off

@register("expr")
class expr(_code):
    """A small block code whose codeword bits are expressions of the message bits: "k", "bits": one expression per
    codeword bit over m0..m{k-1} and the bits before it (c0, c1...), with ^ & | ~ and constants

        {"base": "block", "type": "expr", "k": 4, "bits": ["m0", "m1", "m2", "m3", "m0 ^ m1 ^ m3", "(m1 & m2) ^ 1"]}

    Loaded as a linear code (and its `xor` pattern) when the map is affine over GF(2), so it decodes on its syndrome
    trellis like any; otherwise decoded by nearest codeword over the whole codebook (k <= 16): soft, the codeword whose
    signs best match the LLRs. `errors` is the distance to the codeword chosen (-1 past `t`, when given)"""
    soft = True
    extends = True # "extend" goes into its codebook or matrices

    @classmethod
    def from_config(cls, log, obj):
        k = int(obj["k"])
        book = codebook(obj["bits"], k)
        if not obj.get("nonlinear", False):
            found = affine(book, k)
            if found is not None:
                code = _as_linear(log, *found, bool(obj.get("extend")), obj.get("t"))
                if code is not None:
                    return code
        if obj.get("extend"):
            book = np.concatenate((book, (book.sum(axis=1) % 2)[:, None].astype(np.uint8)), axis=1)
        return cls(log, book, obj.get("t"))

    def __init__(self, log, book, t=None):
        book = np.asarray(book, dtype=np.uint8)
        k = int(np.log2(len(book)))
        super().__init__(log, None, k, book.shape[1])
        self.book, self.t = book, t
        self._signs = 1.0 - 2.0*book.astype(np.float64) # (2^k, n): +1 for a 0
        self._msgs = ((np.arange(1 << k)[:, None] >> np.arange(k - 1, -1, -1)) & 1).astype(bool)

    def encode(self, bits):
        u = np.asarray(bits).astype(np.int64).reshape(-1, self.num)
        idx = u @ (1 << np.arange(self.num - 1, -1, -1))
        return self.book[idx].reshape(-1).astype(bool)

    def _decode(self, bits, soft=None):
        llr = as_llr(bits, soft)
        w = llr[:len(llr)//self.den*self.den].reshape(-1, self.den)
        best = np.argmax(w @ self._signs.T, axis=1)
        hard = (w < 0).astype(np.uint8)
        errs = np.count_nonzero(hard != self.book[best], axis=1).astype(int)
        if self.t is not None:
            errs[errs > int(self.t)] = -1
        return self._msgs[best].reshape(-1), errs

def _as_linear(log, G, off, extend, t):
    """The linear code (in an arrangement: its message bits' positions, the offset) of an affine map, or None when no
    codeword bit is a message bit alone for some message bit (not systematic: decoded from the codebook)"""
    from .layout import arranged
    G = np.asarray(G, dtype=np.uint8)
    k, n = G.shape
    if rank(G) != k:
        raise ValueError("expr code: two messages give the same codeword (the bits do not determine the message)")
    pos = []
    for i in range(k):
        unit = np.flatnonzero((G[i] == 1) & (G.sum(axis=0) == 1))
        if not len(unit):
            return None
        pos.append(int(unit[0]))
    order = pos + [c for c in range(n) if c not in set(pos)] # codeword bit j of the systematic code is sent bit order[j]
    lin = linear.from_config(log, {"n": n, "k": k, "p": G[:, order][:, k:], "extend": extend})
    lin.t = t
    if extend:
        order, off = order + [n], np.append(off, off.sum() % 2)
    identity = order == list(range(len(order)))
    if identity and not off.any():
        return lin
    return arranged(log, lin, None if identity else order, off.astype(int).tolist() if off.any() else None)
