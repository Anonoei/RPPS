"""Kernels: the batch face of an operator, beside the operator it belongs to (rpps.parallel)

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
from collections import namedtuple
from dataclasses import dataclass, field

import numpy as np

# What a block of data is: the class and tags of rpps.Data (CAT, DT, REP) and the rates and delay it carries. A tuple, so it
# is picklable and hashable
Form = namedtuple("Form", "cls cat dt rep Fs sps cf delay")

@dataclass
class Batch:
    """N frames that share a key, as one array. x: [N, L]; form: what the rows are; pos: each row's absolute input position;
    n_in: the frame length they started as; bad: rows that failed, why: {row: reason}; notes: {row: [report, ...]}"""
    x: np.ndarray
    form: Form
    pos: list
    n_in: int
    bad: np.ndarray = None
    why: dict = field(default_factory=dict)
    code_failed: np.ndarray = None # rows the last decoder with a verdict could not decode: dropped when the plan ends
    notes: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.bad is None:
            self.bad = np.zeros(len(self.x), dtype=bool)

    def __len__(self):
        return len(self.x)

    def fail(self, row, why):
        if not self.bad[row]:
            self.bad[row], self.why[row] = True, str(why)

class Kernel:
    """A batch function. Subclasses set `name` and give `run`; `shape` says the output length for an input length (None: not
    known before running; given None, None), `consts` the arrays it was built with"""
    name = "kernel"

    def shape(self, n_in):
        return n_in

    def consts(self):
        return {}

    def run(self, xp, b):
        raise NotImplementedError

    def describe(self, n_in):
        """A JSON-able description: what a backend (a device kernel library, an HLS generator) needs to build it"""
        return {"kernel": self.name, "in": n_in, "out": self.shape(n_in),
                "consts": {k: {"shape": list(np.shape(v)), "dtype": str(np.asarray(v).dtype)} for k, v in self.consts().items()}}

    def __repr__(self):
        return f"<{type(self).__name__} {self.name}>"
