import numpy as np

from ._code import _code, register

@register("forney")
class forney(_code):
    """Forney convolutional interleaver: I branches, branch j delays by j*M symbols (deinterleaver: (I-1-j)*M)

    Works on symbols of `symbol` bits (8: bytes). The first symbol must sit on branch 0 (a packet sync byte).
    The deinterleaver has a start-up of (I-1)*I*M symbols before it carries data.
    """
    @classmethod
    def from_config(cls, log, obj):
        return cls(log, obj["I"], obj["M"], obj.get("symbol", 8))

    def __init__(self, log, I=12, M=17, symbol=8):
        super().__init__(log, None, 1, 1)
        self.I, self.M, self.symbol = I, M, symbol
        self.delay = (I-1)*I*M # Symbols of start-up delay of interleaver + deinterleaver together

    def _to_symbols(self, bits):
        bits = np.asarray(bits).astype(np.uint8)
        bits = bits[:len(bits)//self.symbol*self.symbol]
        return (bits.reshape(-1, self.symbol) * (1 << np.arange(self.symbol-1, -1, -1))).sum(axis=1)

    def _to_bits(self, sym):
        return ((np.asarray(sym)[:, None] >> np.arange(self.symbol-1, -1, -1)) & 1).astype(bool).reshape(-1)

    def _run(self, sym, rev, state=None):
        """state: a dict holding each branch's delay line between calls (sym must then start on branch 0, hold whole rows of I)"""
        out = np.zeros_like(sym)
        for j in range(self.I):
            x = sym[j::self.I]
            d = ((self.I-1-j) if rev else j) * self.M
            line = state.get((rev, j)) if state is not None else None
            if line is None:
                line = np.zeros(d, dtype=sym.dtype)
            both = np.concatenate((line, x))
            out[j::self.I] = both[:len(x)]
            if state is not None:
                state[(rev, j)] = both[len(x):]
        return out

    def encode(self, bits, symbols=False, state=None):
        sym = np.asarray(bits) if symbols else self._to_symbols(bits)
        out = self._run(sym, rev=False, state=state)
        return out if symbols else self._to_bits(out)

    def decode(self, bits, symbols=False, state=None):
        sym = np.asarray(bits) if symbols else self._to_symbols(bits)
        out = self._run(sym, rev=True, state=state)
        return out if symbols else self._to_bits(out)


@register("block")
class block_interleaver(_code):
    """Block interleaver: a fixed permutation of each block of `n` items of `symbol` bits (dibits: symbol=2)

    perm: the permutation, as a list or {"each": "<expression of i>", "n": N} in the definition.
        map "from" (default): item i of the interleaved block is item perm[i] of the input
        map "to": item i of the input goes to position perm[i]
    Or the row-column form (LTE sub-block interleaver, 36.212 5.1.4.2.1):
        cols, col_perm (default: in order), length n (rows = ceil(n/cols)): written row by row with the filler
        (rows*cols - n dummies) at the front, columns permuted, read column by column; dummies are dropped
    encode() interleaves, decode() undoes it, on bits or on soft values (any dtype), whole blocks at a time.
    """
    @classmethod
    def from_config(cls, log, obj):
        sym = obj.get("symbol", 1)
        if "perm" in obj:
            return cls(log, obj["perm"], obj.get("map", "from"), sym)
        return cls(log, rowcol(obj["length"], obj["cols"], obj.get("col_perm"), obj.get("fill", "front")), "from", sym)

    def __init__(self, log, perm, mapping="from", symbol=1):
        perm = np.asarray(perm, dtype=np.int64)
        n = len(perm)
        if not np.array_equal(np.sort(perm), np.arange(n)):
            raise ValueError(f"Not a permutation of 0..{n-1} (repeated or missing positions)")
        if mapping not in ("from", "to"):
            raise ValueError(f"map {mapping!r}: 'from' or 'to'")
        src = perm if mapping == "from" else np.argsort(perm)
        super().__init__(log, None, n*symbol, n*symbol)
        self.n, self.symbol = n, symbol
        self.src = src                       # Output item i = input item src[i]
        self.dst = np.argsort(src)           # Input item j lands at dst[j]
        self.block_bits = n*symbol

    def _apply(self, x, index):
        x = np.asarray(x)
        per = self.block_bits
        x = x[:len(x)//per*per].reshape(-1, self.n, self.symbol)
        return x[:, index, :].reshape(-1)

    def encode(self, bits):
        return self._apply(bits, self.src)

    def decode(self, bits):
        return self._apply(bits, self.dst)

def rowcol(length, cols, col_perm=None, fill="front"):
    """`from` permutation of a row-column interleaver: written by rows (dummies first, or last with fill="back"),
    columns permuted by col_perm, read by columns; dummy positions dropped"""
    rows = -(-length//cols)
    dummies = rows*cols - length
    y = np.full(rows*cols, -1)
    if fill == "front":
        y[dummies:] = np.arange(length)
    else:
        y[:length] = np.arange(length)
    mat = y.reshape(rows, cols)
    order = np.arange(cols) if col_perm is None else np.asarray(col_perm)
    out = mat[:, order].T.reshape(-1)
    return out[out >= 0]
