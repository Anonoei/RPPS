"""Conditioning: DC removal, FM de-emphasis, spectral inversion"""
from types import SimpleNamespace

import numpy as np

from ..base import Stage
from .filters.iir import IIR, single_pole

class DCBlock(Stage):
    """Removes the DC offset (a one pole high pass: y[n] = x[n] - x[n-1] + r*y[n-1]), an `IIR` section"""
    __slots__ = ("r", "_iir")
    def __init__(self, r=0.999):
        self.r = r
        self._iir = IIR([[1.0, -1.0, 0.0, 1.0, -r, 0.0]])

    def new_state(self, meta):
        return self._iir.new_state(meta)

    def step(self, samples, st, meta):
        return self._iir.step(samples, st, meta)


class Deemphasis(Stage):
    """FM broadcast de-emphasis: a one pole low pass with time constant tau (75 us in the Americas, 50 us elsewhere), an
    `IIR` section at the stream's rate"""
    __slots__ = ("tau",)
    def __init__(self, tau=75e-6):
        self.tau = tau

    def new_state(self, meta):
        f = IIR(single_pole(1 - np.exp(-1/(self.tau*meta.Fs))))
        return SimpleNamespace(iir=f, z=f.new_state(meta).z)

    def step(self, samples, st, meta):
        return st.iir._run(samples, st.z)


class Conjugate(Stage):
    """Spectral inversion of a complex signal: x -> conj(x), so a capture whose I and Q were swapped (or taken from the
    lower sideband of a mixer) comes out the right way round. Inverting the spectrum at the symbol level instead is a
    constellation map ("R090 inverted"); this does it on the samples, before any sync"""
    def step(self, samples, st, meta):
        x = np.asarray(samples)
        return np.conj(x) if x.dtype.kind == "c" else x
