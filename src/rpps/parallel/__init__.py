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
  `Gather`), `code.kernels` (`BlockDecode`, `LdpcDecode`, `BchDecode`, `Permute` of an interleaver, `Identity` of a code a
  frame switches off, `XorMask`), `sync.kernels` (`PilotLine`), `link.kernels` (`HeaderDecode` of a framed header); the
  protocol is `rpps.base.kernel`.
- Kernels belong to generic operators, never a protocol; they use only what numpy and cupy share (`xp`), hold their
  constants, keep shapes fixed per key, and mark a failed row instead of raising. `BatchPlan.describe()` is data, what a
  device library or HLS generator would build from.
- Processes use `fork` where available, else `spawn` (scripts need a `__main__` guard); each builds its region from a
  picklable `RegionSteps`.
- On `dvbs2/normal` the output equals the serial runner's bit for bit, for any block cut, worker count and batch size.
  Measure with `python tools/parallel_bench.py 64 4`.
- A parallel region's kernels decode inside: each frame result is its message (`"frame"`, `crc_ok` False when dropped),
  without the header fields a serial runner gives.

## Devices (GPUs)

Vendor neutral: CUDA and ROCm/HIP alike, through CuPy (built for either, one API); nothing here names a vendor, and no
kernel branches on one. `rpps._accel.device()` gives `(xp, "cuda" | "hip")` or `(numpy, None)`;
`Parallel("thread", xp=xp)` runs the region's plans on it (inline or thread executors only: a device does not cross a
fork).

- **Hand-off.** A batch is stacked on the host and the plan moves it: one upload at its start, one download at its end,
  and a `to:host` / `to:device` pair around each run of kernels that need numpy (`Kernel.where == "host"`: the
  decoders, a `LoopKernel`). Every crossing is a kernel, so `BatchPlan.describe()` shows the cost; nothing converts
  implicitly. Constants go to a device once (`Kernel.const`). `bad`, `why`, `notes` and `code_failed` stay on the host.
- **Instructions**, each with the numpy run as its oracle: (1) array code over `xp`, which most kernels are and which
  runs on CuPy unchanged; (2) hand-written device kernels for what array code does badly, in a portable subset of CUDA C
  that HIP compiles as it is (no warp intrinsics, no wavefront width assumed), compiled at run time by CuPy (NVRTC or
  hiprtc) through `rpps.base.kernel.device_functions`: the layered min-sum LDPC decoder (`rpps.code.kernels.LdpcDecode`,
  a launch per layer over every check and codeword, codewords the fastest axis so accesses are contiguous, the batch
  compacted as codewords converge), BCH syndromes (`BchDecode`: only the words with errors go to the host to be located,
  their bits flipped on the device) and the soft demapper (`Demap`, a thread a symbol); (3) `describe()`, the export for
  an FPGA or HLS flow. A header's fields are read on the host from its few bits (`HeaderDecode`); its payload is cut on
  the device. On `dvbs2/normal` the whole region stays on a device: one upload, one download.
- **Getting results back.** The download at the plan's end gives numpy; results are built on the host and the
  `Scheduler` puts them in order, as for any executor. A dedicated device executor (later) overlaps the next batch's
  upload with this one's work on device streams, with `Scheduler(eager=False)` so batches fill.
- **Checked without a GPU.** `tests/unit/fakedevice.py` is numpy behind an array type that refuses what CuPy refuses
  (no implicit conversion, no numpy array mixed in); every kernel and the DVB-S2 region run on it in the suite.
- **Batches pay on a device.** A layer launch needs thousands of (check, codeword) threads to fill a GPU, so a device
  wins from about 16 frames a batch (`Parallel(max_batch=)`) and is even at one; measure with `tools/device_bench.py`.
  On an AMD Radeon 860M (an integrated GPU) against 16 CPU threads with the C++ kernels, 16-64 frames a batch: LDPC
  2.1-2.6x, BCH 2.7-3.8x (clean words; about 1.1x with five errors a word), demap 4.5-7.9x; the DVB-S2 region 1.9x,
  and the receiver end to end 1.35x, the serial front now most of its time.
- **What moves next:** the front's block estimators (matched filter, timing, carrier, frame sync); its per-sample
  loops stay on the CPU.
"""
from .frame import Frame, Result, OK, DROPPED
from .executor import Executor, InlineExecutor, ThreadExecutor, ProcessExecutor, make
from .scheduler import Scheduler
from .kernel import LoopKernel
from .plan import BatchPlan, compile
from .worker import PlanWorker
from .runner import Parallel, ParallelRunner
