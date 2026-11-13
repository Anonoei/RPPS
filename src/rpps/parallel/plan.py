"""Batch plans: a region's kernels for one set of parameters (README.dev.md, Parallelism)

A `BatchPlan` is what a worker compiles for a `Frame.key` (the parameters the front resolved, and the frame's shape) and runs
on every batch with that key. It chains the region's kernels, checks the lengths between them where they are known, keeps the
time each spent, and describes itself in a form a device backend can build (`describe`).
"""
import time

from ..errors import ConfigError
from ..profile import control as _ctl
from .kernel import kernel_of

class BatchPlan:
    def __init__(self, kernels, key=None):
        self.kernels, self.key = list(kernels), key
        self.spent = {k.name: 0.0 for k in self.kernels}
        self.batches = 0

    def run(self, xp, b):
        for k in self.kernels:
            t0 = time.perf_counter()
            b = k.run(xp, b)
            self.spent[k.name] += time.perf_counter() - t0
        self.batches += 1
        return b

    def describe(self, n_in):
        """[{kernel, in, out, consts}] from a frame of `n_in` items; a length a kernel can't tell before running is None"""
        out, n = [], n_in
        for k in self.kernels:
            d = k.describe(n)
            out.append(d)
            n = d["out"]
        return out

    def __repr__(self):
        return f"<BatchPlan " + " > ".join(k.name for k in self.kernels) + ">"

def compile(steps, params, direction="rx", key=None):
    """The plan of `steps` (built steps, in order) for `params` ({step id: {parameter: value}}). A parameter a step takes live
    is applied to every frame as its start; one it takes by restart builds the step again with it (what a header did to
    a runner)"""
    kernels = []
    for st in steps:
        p = params.get(st.id, {})
        live = {k: v for k, v in p.items() if st.controls.get(k) == _ctl.LIVE}
        restart = {k: v for k, v in p.items() if st.controls.get(k) == _ctl.RESTART}
        if restart:
            new = st.rebuild(restart)
            new.takes, new.gives = st.takes, st.gives
            st = new
        for k in p:
            if k not in st.controls:
                raise ConfigError(f"step {st.id!r} cannot be controlled on {k!r} (it takes {sorted(st.controls)})")
        kernels.append(kernel_of(st, live, direction))
    return BatchPlan(kernels, key)
