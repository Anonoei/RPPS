from types import SimpleNamespace

import numpy as np

from .. import _accel
from ..base import Meta, Stage
from ..base.types import Category
from ..sequence.code import Code

def correlate(x, code: Code, sps=1):
    """Correlate whole code periods of x (len a multiple of code.length*sps) with every sequence of the code: (periods, code.count).
    Each correlation is normalised, so a signal that is the sequence times a symbol gives that symbol"""
    x = np.asarray(x)
    if x.dtype not in (np.complex64, np.complex128):
        x = x.astype(np.complex128)
    sps = int(sps)
    if len(x) % (code.length*sps):
        raise ValueError(f"{len(x)} samples is not a whole number of {code.length} chip periods at {sps} samples/chip")
    if _accel.native is not None:
        return _accel.native.despread(x, np.ascontiguousarray(code.chips), sps)
    chip = x.reshape(-1, code.length, sps).sum(axis=2) # Samples of each chip, summed
    return (chip @ code.chips.T / (code.length*sps)).astype(x.dtype)

class Despread(Stage):
    """Despread a direct-sequence signal: one value per code period, the correlation with the code

    The input is at `sps` samples per chip (from the stream's sps, or the argument), aligned to the code: the first
    `offset` samples (a code phase from acquisition) are skipped, then every N chips are one period. Output rates:
    Fs / (N*sps), 1 sample per symbol.

    code: a Code. index: which of its sequences to correlate with (a set has more than one; for the strongest of a set
    use `correlate` and take the argmax). offset: samples to skip at the start of the stream. sps: samples per chip
    """
    CATEGORY = Category.SYMBOL # The stage form of Spread.decode

    def __init__(self, code: Code, index=0, offset=0, sps=None):
        if not 0 <= int(index) < code.count:
            raise ValueError(f"index {index}: the code has {code.count} sequences")
        self.code = code
        self.index = int(index)
        self.offset = int(offset)
        self.sps = None if sps is None else int(sps)
        self._one = Code(code.name, code.chips[self.index], code.rate)

    def new_state(self, meta: Meta):
        return SimpleNamespace(skip=self.offset, buf=None)

    def step(self, samples, st, meta: Meta):
        sps = self.sps if self.sps is not None else meta.isps
        if sps < 1:
            raise ValueError(f"Despread needs samples per chip >= 1, got {sps}: pass sps=")
        N = self._one.length * sps
        x = np.asarray(samples)
        if st.skip:
            cut = min(st.skip, len(x))
            x, st.skip = x[cut:], st.skip - cut
        if st.buf is not None and len(st.buf):
            x = np.concatenate((st.buf, x))
        n = len(x)//N*N
        st.buf = x[n:].copy()
        meta.op.Fs = meta.Fs / N
        meta.op.sps = 1
        if n == 0:
            return x[:0].astype(np.complex64 if x.dtype != np.complex128 else np.complex128)
        return correlate(x[:n], self._one, sps)[:, 0]
