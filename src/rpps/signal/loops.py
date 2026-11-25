"""Conditioning as per-sample loops (second class): the recursions `DCBlock` and `Deemphasis` run sample by sample, in
the C++ kernels when they are built (rpps._native: dc_block, one_pole) and a Python loop otherwise

The first-class stages, rpps.signal.DCBlock and rpps.signal.Deemphasis, are the same recursions as IIR sections run with
no loop over the samples (rpps.signal.filters.iir); these give the same output to rounding. They stay as the references
the sections are checked against, and for whoever wants the plain loop.
"""
from types import SimpleNamespace

import numpy as np

from .. import _accel
from ..base import Stage

def _double(x):
    x = np.asarray(x)
    return x.astype(np.complex128 if x.dtype.kind == "c" else np.float64)

class DCBlock(Stage):
    """Removes the DC offset, a one pole high pass run sample by sample: y[n] = x[n] - x[n-1] + r*y[n-1]"""
    __slots__ = ("r",)
    def __init__(self, r=0.999):
        self.r = r

    def new_state(self, meta):
        return SimpleNamespace(x=0.0, y=0.0)

    def step(self, samples, st, meta):
        x = np.ascontiguousarray(samples)
        if _accel.native is not None and x.dtype in _accel.NATIVE_DTYPES:
            cplx = x.dtype.kind == "c"
            out, st.x, st.y = _accel.native.dc_block(x, complex(st.x) if cplx else float(np.real(st.x)),
                                                     complex(st.y) if cplx else float(np.real(st.y)), float(self.r))
            return out
        out = np.empty_like(samples)
        x1, y1, r = st.x, st.y, self.r
        for i, v in enumerate(_double(samples)): # (in double, as the kernel)
            y1 = v - x1 + r*y1
            x1 = v
            out[i] = y1
        st.x, st.y = x1, y1
        return out


class Deemphasis(Stage):
    """FM broadcast de-emphasis, a one pole low pass with time constant tau run sample by sample: y += a*(x - y)"""
    __slots__ = ("tau",)
    def __init__(self, tau=75e-6):
        self.tau = tau

    def new_state(self, meta):
        return SimpleNamespace(y=0.0)

    def step(self, samples, st, meta):
        a = 1 - np.exp(-1/(self.tau*meta.Fs))
        x = np.ascontiguousarray(samples)
        if _accel.native is not None and x.dtype in _accel.NATIVE_DTYPES:
            out, st.y = _accel.native.one_pole(x, complex(st.y) if x.dtype.kind == "c" else float(np.real(st.y)), float(a))
            return out
        out = np.empty_like(samples)
        y = st.y
        for i, v in enumerate(_double(samples)):
            y += a*(v - y)
            out[i] = y
        st.y = y
        return out
