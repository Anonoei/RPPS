"""Batch kernels of the sync operators: `PilotLine` of `PilotPhase`

Checked row for row against the operator run frame by frame (tests/unit/test_batch_kernels.py); they use only what
numpy and cupy share, so `xp` can be either (rpps.parallel)
"""
import numpy as np

from ..base.data import Analog
from ..base.kernel import Form, Kernel
from ..base.types import Category, Rep, dtype


class PilotLine(Kernel):
    """The carrier phase of each frame from its pilot blocks, taken off: a least-squares line through the phase of every block (and
    through `lead`, a position where it is known to be 0), per row. Symbols past the frame pass unchanged, a frame cut short
    as it is. `starts`: where the blocks start in a frame, `total`: its length with the pilots"""
    def __init__(self, starts, total, length, value, lead, name="pilot_line"):
        self.starts, self.total, self.length, self.value, self.lead = np.asarray(starts, np.int64), int(total), int(length), complex(value), lead
        self.name = f"line:{name}"
        pos = [float(s) + self.length/2 for s in self.starts]
        self.pos = np.array(([float(lead)] if lead is not None else []) + pos)

    def shape(self, n_in):
        return n_in

    def consts(self):
        return {"starts": self.starts, "pos": self.pos}

    def run(self, xp, b):
        N, L = b.x.shape
        b.form = Form(Analog, Category.SYNC, dtype.SYMBOLS, Rep.SYMBOLS, *b.form[4:]) # (what a stage gives)
        if L < self.total or not len(self.pos):
            b.x = b.x.astype(np.complex64) # (a frame cut short is left as it came; with nothing to measure the symbols pass)
            return b
        Y = b.x[:, :self.total].astype(np.complex128)
        ph = xp.zeros((N, len(self.pos)))
        k = 1 if self.lead is not None else 0
        if len(self.starts):
            idx = self.const(xp, "idx", lambda: self.starts[:, None] + np.arange(self.length)[None, :])
            blocks = Y[:, idx]                                                   # (N, blocks, length)
            ph[:, k:] = xp.angle((blocks*np.conj(self.value)).mean(axis=2))
        ph = xp.unwrap(ph, axis=1)
        if len(self.pos) > 1:
            d = self.pos - self.pos.mean()
            slope = (ph*self.const(xp, "d", lambda: d)[None, :]).sum(axis=1)/float((d*d).sum())
            icpt = ph.mean(axis=1) - slope*float(self.pos.mean())
        else:
            slope, icpt = xp.zeros(N), ph[:, 0]
        turn = icpt[:, None] + slope[:, None]*xp.arange(self.total)[None, :]
        out = b.x.astype(np.complex128)
        out[:, :self.total] = Y*xp.exp(-1j*turn)
        b.x = out.astype(np.complex64)
        return b
