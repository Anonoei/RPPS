from types import SimpleNamespace

import numpy as np

from .sync import Sync, Meta

def _o2(sym):
    return np.real(sym) * np.imag(sym) / (np.real(sym)**2 + np.imag(sym)**2 + 1e-12)

def _o4(sym):
    a = 1.0 if sym.real > 0 else -1.0
    b = 1.0 if sym.imag > 0 else -1.0
    return (a * sym.imag - b * sym.real) / (abs(sym) + 1e-12)

class Costas:
    """Costas loop. Settings live here, `freq` and `pha` live in `meta.state_of(costas)`

    The error is normalised by the symbol amplitude, so alpha/beta don't depend on signal level
    """
    __slots__ = (
        "alpha", "beta",
        "err_func"
    )
    def __init__(self, alpha, beta, order=2):
        self.alpha = alpha
        self.beta = beta
        if order == 2:
            self.err_func = _o2
        elif order == 4:
            self.err_func = _o4
        else:
            raise ValueError(f"Costas order must be 2 or 4, got {order}")

    def new_state(self, meta: Meta):
        return SimpleNamespace(pha=0.0, freq=0.0)

    def run(self, sym, st):
        out = sym * np.exp(-1j*st.pha)
        error = self.err_func(out)

        st.freq += (self.beta * error)
        st.pha += st.freq + (self.alpha * error)

        while st.pha >= 2*np.pi:
            st.pha -= 2*np.pi
        while st.pha < 0:
            st.pha += 2*np.pi
        return out

    def burst(self, meta: Meta):
        st = meta.state_of(self)
        out = np.zeros(len(meta.obj), dtype=meta.obj.dtype)
        for i, sym in enumerate(meta.obj):
            out[i] = self.run(sym, st)
        meta.obj = out
        return meta
