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
