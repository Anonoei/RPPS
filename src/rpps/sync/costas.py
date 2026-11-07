from types import SimpleNamespace

import numpy as np

from ..base import Meta, Stage

def _o2(sym):
    return np.real(sym) * np.imag(sym) / (np.real(sym)**2 + np.imag(sym)**2 + 1e-12)

def _o4(sym):
    a = 1.0 if sym.real > 0 else -1.0
    b = 1.0 if sym.imag > 0 else -1.0
    return (a * sym.imag - b * sym.real) / (abs(sym) + 1e-12)

def _oM(M, offset):
    """Phase error of M-PSK symbols with a point at `offset` radians: distance to the nearest decision, in radians"""
    step = 2*np.pi/M
    def err(sym):
        ang = np.angle(sym) - offset
        return (ang - np.round(ang/step)*step)
    return err

class Costas(Stage):
    """Costas loop. Settings live here, `freq` and `pha` live in `meta.state_of(costas)`

    The error is normalized by the symbol amplitude, so alpha/beta don't depend on signal level
    """
    __slots__ = (
        "alpha", "beta",
        "err_func"
    )
    def __init__(self, alpha, beta, order=2, offset=0.0):
        """alpha, beta: proportional and integral gain (see `from_bandwidth`)
        order: 2 (BPSK), 4 (QPSK, points at 45 degrees) or any M >= 3 for M-PSK with a point at `offset` radians"""
        self.alpha = alpha
        self.beta = beta
        if order == 2:
            self.err_func = _o2
        elif order == 4:
            self.err_func = _o4
        elif order >= 3:
            self.err_func = _oM(order, offset)
        else:
            raise ValueError(f"Costas order must be 2 or more, got {order}")

    @classmethod
    def from_bandwidth(cls, bw, order=2, damping=0.7071, offset=0.0):
        """Loop gains from a noise bandwidth normalised to the symbol rate (0.01-0.05 typical)"""
        from .pll import loop_gains
        return cls(*loop_gains(bw, damping), order=order, offset=offset)

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

    def step(self, samples, st, meta: Meta):
        out = np.zeros(len(samples), dtype=np.result_type(samples.dtype, np.complex64))
        for i, sym in enumerate(samples):
            out[i] = self.run(sym, st)
        return out
