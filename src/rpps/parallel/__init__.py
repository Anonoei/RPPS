"""Parallel processing of frames (ARCHITECTURE.md, Parallelism; Migration step 10)

The serial front of a receiver finds frames and decides their parameters; a parallel region then takes each frame as a
self-contained `Frame`, and a `Scheduler` batches them to an `Executor` and puts the `Result`s back in order. See
`rpps.profile.parallel` for how a profile marks where the region starts.
"""
from .frame import Frame, Result, Form, OK, DROPPED
from .executor import Executor, InlineExecutor, ThreadExecutor, ProcessExecutor, make
from .scheduler import Scheduler
from .kernel import Batch, Kernel, LoopKernel
from .plan import BatchPlan, compile
from .worker import PlanWorker
from .runner import Parallel, ParallelRunner
