"""Pulse shaping and its matched filter: the wave pair of a symbol pulse (README.dev.md, Wave pairs)

The pulses themselves are definitions (config/pulse, built by rp.signal.filters.load("pulse/...")); these apply them:

    Shape(sps, pulse)   transmit: `sps` samples a symbol, each symbol carrying the pulse (zero stuffing and the pulse, in
                        polyphase form, so the zeros are never multiplied: the same output as `Zeros(sps) + pulse`)
    Matched(pulse)      receive: the pulse time-reversed and conjugated, scaled so that pulse and matched filter give a
                        symbol back at the amplitude it was sent (unit=False: the taps as they are, an integrate-and-dump)

It does not choose the sampling instant: that is timing recovery (rpps.sync.time).
"""
from types import SimpleNamespace

import numpy as np

from ..signal.filters.filter import FIR
from ..signal.sample.up import Interpolate

class Shape(Interpolate):
    """Symbols to `sps` samples a symbol, each carrying `pulse` (an FIR stage, a pulse definition)"""
    __slots__ = ()
    def __init__(self, sps, pulse):
        super().__init__(int(sps), pulse)

    def __str__(self):
        return f"Shape({self.ratio}, {self.filt})"

class Matched(FIR):
    """The matched filter of `pulse` (an FIR stage): its taps reversed and conjugated, divided by sum |h|^2 when `unit`
    (the peak of pulse * matched filter), so a symbol of 1 comes back as 1"""
    __slots__ = ("pulse", "unit")
    def __init__(self, pulse, unit=True):
        self.pulse, self.unit = pulse, bool(unit)

    def __str__(self):
        return f"Matched({self.pulse})"

    def make_taps(self, meta):
        return np.conj(np.asarray(self.pulse.new_state(meta).taps)[::-1])

    def new_state(self, meta):
        taps = np.asarray(self.make_taps(meta))
        if self.unit:
            taps = taps/np.sum(np.abs(taps)**2)
        if taps.dtype.kind == "c" and not np.any(taps.imag):
            taps = taps.real
        return SimpleNamespace(taps=taps, hist=None)
