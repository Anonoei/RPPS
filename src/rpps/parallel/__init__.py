"""Parallel processing of frames: a profile's frame-parallel region

Real time needs more than one core, and later a GPU or FPGA. Once frame sync has found a frame and its header has set
the parameters, the rest is per-frame work with no state across frames. A profile marks where that region starts
(`"parallel": "<step id>"`); `rpps.profile.parallel` refuses a region with stream-scoped steps (`SCOPE = STREAM`), a
publishing header or a late change.

The unit is the **batch**, not the thread: N frames through one program pays on every target.

    profile.rx(parallel=rp.parallel.Parallel("thread", workers=4))   -> ParallelRunner
      front       the ordinary Runner, stopped at the region; the changes it resolved cut the input at frame starts
      Scheduler   frames -> batches of one key -> Executor (inline, threads by rows, processes, later GPU / FPGA) -> results in order
      BatchPlan   per key (the header's parameters, the frame's shape): the region's kernels, shapes checked, describe()
      Kernel      a pure function over [N, L] arrays, run(xp, batch); LoopKernel runs frame by frame through the operators

- An operator gives its kernel with `op.kernel(params, how)`; a step without one runs through `LoopKernel`, which is also
  the oracle every kernel is checked against (`test_batch_kernels.py`). Kernels: `symbol.kernels` (`Demap`, `Rotate`,
  `Gather`), `code.kernels` (`BlockDecode`, `XorMask`), `sync.kernels` (`PilotLine`); the protocol is `rpps.base.kernel`.
- Kernels belong to generic operators, never a protocol; they use only what numpy and cupy share (`xp`), hold their
  constants, keep shapes fixed per key, and mark a failed row instead of raising. `BatchPlan.describe()` is data, what a
  device library or HLS generator would build from.
- Processes use `fork` where available, else `spawn` (scripts need a `__main__` guard); each builds its region from a
  picklable `RegionSteps`.
- On `dvbs2/normal` the output equals the serial runner's bit for bit, for any block cut, worker count and batch size.
  Measure with `python tools/parallel_bench.py 64 4`.
- A parallel region's kernels decode inside: each frame result is its message (`"frame"`, `crc_ok` False when dropped),
  without the header fields a serial runner gives.
"""
from .frame import Frame, Result, OK, DROPPED
from .executor import Executor, InlineExecutor, ThreadExecutor, ProcessExecutor, make
from .scheduler import Scheduler
from .kernel import LoopKernel
from .plan import BatchPlan, compile
from .worker import PlanWorker
from .runner import Parallel, ParallelRunner
