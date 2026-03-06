import numpy as np

from .sync import Sync, Meta

def _o2(sym):
    return np.real(sym) * np.imag(sym)

def _o4(sym):
    a = 1.0 if sym.real > 0 else -1.0
    b = 1.0 if sym.imag > 0 else -1.0
    return a * sym.imag - b * sym.real

class Costas:
    __slots__ = (
        "alpha", "beta",
        "err_func",
        "freq", "pha"
    )
    def __init__(self, alpha, beta, order=2):
        self.alpha = alpha
        self.beta = beta
        if order == 2:
            self.err_func = _o2
        elif order == 4:
            self.err_func = _o4

        self.pha = 0.0
        self.freq = 0.0

    def run(self, sym):
        out = sym * np.exp(-1j*self.pha)
        error = self.err_func(out)

        self.freq += (self.beta * error)
        self.pha += self.freq + (self.alpha * error)

        while self.pha >= 2*np.pi:
            self.pha -= 2*np.pi
        while self.pha < 0:
            self.pha += 2*np.pi
        return out

    def burst(self, meta: Meta):
        out = np.zeros(len(meta.obj), dtype=meta.obj.dtype)
        for i, sym in enumerate(meta.obj):
            out[i] = self.run(sym)
        meta.obj = out
        return meta
