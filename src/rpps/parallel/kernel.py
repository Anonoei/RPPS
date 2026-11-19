"""Running a step on a batch when its operators give no kernel (the protocol is rpps.base.kernel; README.dev.md, Parallelism)

A region frame is whole, so what an operator does to it is a function of the frame and of the parameters controls gave it: no
state across calls. A `Kernel` is that function over a batch of frames, one frame per row of an `[N, L]` array:

    batch = kernel.run(xp, batch)

`xp` is the array module (numpy; cupy on a GPU), so a kernel is written once. A kernel is immutable once built and holds its
constants (a keystream, a permutation, a code's graph), made once for its parameters, so they can live on a device and be
exported (`describe`). Rows that cannot be processed are marked in `batch.bad` (with a reason) and carry zeros on: the shape of
a batch never changes with the data.

An operator gives one with `op.kernel(params, how)` (params: its controlled parameters; how: `/` decode, `*` encode, `+` stage;
returns None for what it has no kernel for). A step whose operators do not is run by `LoopKernel`: the existing
per-frame operator, through `Data`. That path is also the oracle every kernel is compared with.
"""
from dataclasses import dataclass, field

import numpy as np

from ..base.data import _Data
from ..base.meta import Meta, Stream
from ..base.rpps import FLUSH, DROPPED
from ..errors import RPPSError, StreamError
from ..profile.steps import STAGE, DECODE
from ..base.kernel import Form, Batch, Kernel

class LoopKernel(Kernel):
    """A step run one frame at a time through its operators (`Data`, a fresh stream per frame): correct for any operator, and
    the reference the batch kernels are checked against. `live`: the controlled parameters applied to each frame as its start"""
    def __init__(self, step, live, direction="rx"):
        self.step, self.live, self.direction = step, live, direction
        self.name = f"loop:{step.id}"

    def shape(self, n_in):
        return None

    def _ops(self):
        return self.step.rx if self.direction == "rx" else self.step.tx

    def _row(self, b, i):
        cls, cat, dt, rep, Fs, sps, cf, delay = b.form
        for _, op in self._ops():
            if self.live and hasattr(op, "control"):
                op.control(**self.live) # a frame starts here
        meta = Meta(Fs, sps, cf, delay, Stream(Fs), b.pos[i])
        meta.flush = True
        d = cls(b.x[i], cat, dt, {FLUSH: True}, meta, rep)
        for how, op in self._ops():
            d = d + op if how == STAGE else d / op if how == DECODE else d * op
        return d

    def run(self, xp, b):
        n = len(b)
        outs = [None]*n
        for i in range(n):
            if b.bad[i]:
                continue
            try:
                outs[i] = self._row(b, i)
            except RPPSError as e: # a frame that cannot be decoded is lost, not fatal
                b.fail(i, f"{type(e).__name__}: {e}")
        ok = [d for d in outs if d is not None]
        if not ok:
            b.x = np.zeros((n, 0), dtype=b.x.dtype)
            return b
        arrs = [np.asarray(d.data) for d in ok]
        if len({len(a) for a in arrs}) != 1:
            raise StreamError(f"{self.name}: frames of one batch gave different lengths {sorted({len(a) for a in arrs})}")
        y = np.zeros((n, len(arrs[0])), dtype=arrs[0].dtype)
        for d, a in zip(ok, arrs):
            y[outs.index(d)] = a
        for i, d in enumerate(outs):
            if d is not None:
                for k in d.ctx.get(DROPPED, []):
                    b.notes.setdefault(i, []).append({"what": "cut short", "items": int(k), "at": int(b.pos[i] + b.n_in)})
        d0 = ok[0]
        b.x, b.form = y, Form(type(d0), d0.CAT, d0.DT, d0.REP, *b.form[4:])
        return b

def kernel_of(step, live, direction="rx"):
    """The kernel of a step with these controlled parameters: the operator's own (`op.kernel(params, how)`) when the step is one
    operator that has one, else a `LoopKernel`"""
    ops = step.rx if direction == "rx" else step.tx
    if len(ops) == 1 and hasattr(ops[0][1], "kernel"):
        how, op = ops[0]
        k = op.kernel(live, how)
        if k is not None:
            return k
    return LoopKernel(step, live, direction)
