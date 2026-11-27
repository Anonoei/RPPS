"""Kernels: the batch face of an operator, beside the operator it belongs to (rpps.parallel)

A region frame is whole, so what an operator does to it is a function of the frame and of the parameters controls gave it: no
state across calls. A `Kernel` is that function over a batch of frames, one frame per row of an `[N, L]` array:

    batch = kernel.run(xp, batch)

`xp` is the array module (numpy; cupy on a GPU), so a kernel is written once. A kernel is immutable once built and holds its
constants (a keystream, a permutation, a code's graph), made once for its parameters, so they can live on a device and be
exported (`describe`). A constant goes to a device once: `self.const(xp, key, make)` keeps what `make()` gives, as an
array of `xp`. Rows that cannot be processed are marked in `batch.bad` (with a reason) and carry zeros on: the shape of
a batch never changes with the data.

Where a batch lives: `b.x` is an array of `xp` (on a device, a device array); `bad`, `why`, `code_failed`, `notes` and
`pos` stay on the host (small). A kernel that needs numpy says so (`where = "host"`, a C++ decoder, a `LoopKernel`): a
plan on another array module puts a `ToHost` before it and a `ToDevice` after, so every crossing is in `describe()`.
Nothing converts implicitly: `to_host(a)` is the one way back.

Most kernels are array code over `xp`. A loop that array code does badly (LDPC's layers, the demapper's sums over points)
is a device kernel of its own: portable CUDA C that NVRTC and hiprtc compile as it is, built by `device_functions` when
`has_device_kernels(xp)`; on numpy the kernel runs its host path, the reference.

An operator gives one with `op.kernel(params, how)` (params: its controlled parameters; how: `/` decode, `*` encode, `+` stage;
returns None for what it has no kernel for). A step whose operators do not is run by `LoopKernel`: the existing
per-frame operator, through `Data`. That path is also the oracle every kernel is compared with.
"""
import threading
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

def to_host(a):
    """A numpy array of `a` (a device array's copy back, `a.get()`; a numpy array as it is)"""
    if type(a) is np.ndarray:
        return a
    get = getattr(a, "get", None)
    return get() if get is not None else np.asarray(a)

def has_device_kernels(xp):
    """True when `xp` compiles and launches kernels of its own (CuPy, on CUDA or HIP); not numpy, not a pretend device"""
    return xp is not np and hasattr(xp, "RawModule") and hasattr(xp, "cuda")

_MODULES, _MODULES_LOCK = {}, threading.Lock()

def device_functions(xp, source, names):
    """The functions `names` of the device kernel `source` (portable CUDA C: NVRTC or hiprtc compile it as it is), compiled
    once per process, device and source"""
    key = (xp.__name__, xp.cuda.Device().id, source)
    with _MODULES_LOCK:
        f = _MODULES.get(key)
        if f is None:
            mod = xp.RawModule(code=source)
            f = _MODULES[key] = {n: mod.get_function(n) for n in names}
    return f

class Kernel:
    """A batch function. Subclasses set `name` and give `run`; `shape` says the output length for an input length (None: not
    known before running; given None, None), `consts` the arrays it was built with. where: "any" (runs on the `xp` it is
    given) or "host" (numpy only)"""
    name = "kernel"
    where = "any"

    def const(self, xp, key, make):
        """`make()` as an array of `xp`, made and moved once per array module and key (a constant stays on its device)"""
        cache = self.__dict__.setdefault("_consts", {})
        k = (xp.__name__ if hasattr(xp, "__name__") else id(xp), key)
        v = cache.get(k)
        if v is None:
            v = cache[k] = xp.asarray(make())
        return v

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

class ToHost(Kernel):
    """The batch back to the host (before a kernel that needs numpy)"""
    name = "to:host"
    def run(self, xp, b):
        b.x = to_host(b.x)
        return b

class ToDevice(Kernel):
    """The batch onto the device of `xp` (after a kernel that ran on the host)"""
    name = "to:device"
    def run(self, xp, b):
        b.x = xp.asarray(b.x)
        return b
