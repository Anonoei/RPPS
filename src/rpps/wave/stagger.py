from types import SimpleNamespace

import numpy as np

from ..base import Stage

class Stagger(Stage):
    """Offset one rail of a complex signal by a part of a symbol: the stagger of O-QPSK

        tx = symbols + Zeros(sps) + halfsine + Stagger()             # Q half a symbol after I
        rx = samples + Stagger(rail="i") + ...                       # I delayed to line up with Q again

    A carrier phase offset mixes I into Q while Q is still late, so take it off the staggered signal *before* `Stagger(rail="i")`
    (a loop that works on the aligned symbols is too late).

    rail: which of "q" (imaginary part) or "i" (real part) is delayed. delay: symbols, 0.5 for O-QPSK; delay * samples per
    symbol must be a whole number. The delayed rail is zero for the first samples and its last ones stay held in the
    stage: a stream that ends has them still inside (a receiver delaying I loses nothing it needs)
    """
    def __init__(self, rail="q", delay=0.5):
        if rail not in ("i", "q"):
            raise ValueError(f"rail must be 'i' or 'q', not {rail!r}")
        self.rail, self.delay = rail, float(delay)

    def __str__(self):
        return f"Stagger({self.rail})"

    def _samples(self, meta):
        n = self.delay*meta.sps
        if meta.sps < 1 or abs(n - round(n)) > 1e-6:
            raise ValueError(f"Stagger of {self.delay} symbol(s) needs a whole number of samples: {n:g} at {meta.sps:g} samples/symbol")
        return int(round(n))

    def new_state(self, meta):
        return SimpleNamespace(held=np.zeros(self._samples(meta), dtype=np.float32))

    def step(self, samples, st, meta):
        x = np.asarray(samples)
        if x.dtype.kind != "c":
            raise ValueError("Stagger needs a complex signal (I and Q)")
        n = len(st.held)
        rail = x.imag if self.rail == "q" else x.real
        joined = np.concatenate((st.held, rail.astype(np.float32)))
        late, st.held = joined[:len(x)], joined[len(x):].copy()
        if self.rail == "i": # Realigning the rails: the data now comes out `delay` symbols later (transmit timing follows I)
            meta.delay += len(st.held)/meta.Fs
        keep = x.real if self.rail == "q" else x.imag
        out = np.empty(len(x), dtype=np.complex64)
        if self.rail == "q":
            out.real, out.imag = keep, late
        else:
            out.real, out.imag = late, keep
        return out
