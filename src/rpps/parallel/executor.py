"""Executors: where a parallel region runs (ARCHITECTURE.md, Parallelism)

An executor takes batches of frames that share a key and gives back their results. The work itself is a `worker`: a
callable `worker(frames) -> [Result]` (one result per frame, same order), built by the region once per executor, so every
backend runs the same thing:

    InlineExecutor   the caller's thread, one batch at a time: the reference, and what tests compare against
    ThreadExecutor   a pool of threads; pays off where the worker releases the GIL (numpy, the rpps._native kernels)
    ProcessExecutor  a pool of processes, each with its own copy of the worker: what stays Python scales too

A process pool, a GPU or an FPGA are further executors with the same four methods: only what `worker` is (a compiled kernel
group, a bitstream and its DMA) and how batches get there changes. A worker never raises for one bad frame: it returns
`Result.dropped`; an exception is a bug and comes out at `result()`.
"""
import os
import queue
import warnings
from concurrent.futures import Future, ThreadPoolExecutor

from .frame import Result

class Executor:
    """Runs batches of frames. Subclasses set `capacity` (batches in flight before the scheduler waits)"""
    capacity = 1

    def prepare(self, worker):
        """Get ready to run `worker`: build per-thread operators, load a bitstream, compile a kernel graph"""
        self.worker = worker

    def submit(self, batch) -> Future:
        """Start a batch (a list of frames) and return a Future of its list of Results"""
        raise NotImplementedError

    def close(self):
        """Release threads, processes, devices"""

    def __enter__(self):
        return self
    def __exit__(self, *exc):
        self.close()

class InlineExecutor(Executor):
    """Run each batch right here, in order. Identical to calling the worker"""
    capacity = 1

    def submit(self, batch):
        f = Future()
        try:
            f.set_result(_run(self.worker, batch))
        except BaseException as e:
            f.set_exception(e)
        return f

class ThreadExecutor(Executor):
    """A pool of threads; a batch is split across them (min_rows: frames a run needs at least). workers: threads (0: RPPS_THREADS,
    else the CPU count)"""
    def __init__(self, workers=0, capacity=None, min_rows=1):
        n = workers or int(os.environ.get("RPPS_THREADS", 0)) or os.cpu_count() or 1
        self.workers, self.min_rows = n, max(1, int(min_rows))
        self.capacity = capacity or 2*n
        self._pool = ThreadPoolExecutor(n, thread_name_prefix="rpps-frame")

    def prepare(self, worker):
        """worker: a callable, or one with `per_thread()` that makes another: every thread then has its own operators (an
        operator keeps state while it runs). They are made here, so the first frames don't pay for building them"""
        self.worker = worker
        make = getattr(worker, "per_thread", None)
        self._idle = queue.SimpleQueue()
        for _ in range(self.workers):
            self._idle.put(make() if make is not None else worker)

    def submit(self, batch):
        """One batch is split into up to `workers` runs of consecutive frames (of one key, so each is a batch of its own):
        what a kernel does to a batch is large array work that releases the GIL, so the runs go on together"""
        n = min(self.workers, max(1, len(batch)//self.min_rows))
        size = -(-len(batch)//n)
        parts = [batch[i:i + size] for i in range(0, len(batch), size)]
        futs = [self._pool.submit(self._one, part) for part in parts]
        return futs[0] if len(futs) == 1 else _Joined(futs)

    def _one(self, batch):
        w = self._idle.get() # (at most `workers` runs at once, so one is always there)
        try:
            return _run(w, batch)
        finally:
            self._idle.put(w)

    def close(self):
        self._pool.shutdown(wait=True)

_CHILD = None

def _child_init(worker, paths, threads):
    """Runs in each worker process: its own copy of the worker (its operators and plans), the config paths of the parent, and
    a share of the cores for the native kernels"""
    global _CHILD
    import os
    os.environ["RPPS_THREADS"] = str(threads)
    if paths:
        from .. import _config
        for p in paths:
            _config.add_config_path(p)
    _CHILD = worker

def _child_run(batch):
    return _run(_CHILD, batch)

class _Async:
    """The Future of a task given to a multiprocessing Pool"""
    def __init__(self, res):
        self.res = res
    def done(self):
        return self.res.ready()
    def result(self):
        return self.res.get()

class ProcessExecutor(Executor):
    """A pool of worker processes, each with its own copy of the worker (built there from its recipe, so it takes whatever a
    worker pickles to: a `PlanWorker`, a module level function). Frames and results cross as pickles; a batch is split across
    the processes by rows, as in `ThreadExecutor`. No GIL is shared, so what stays Python in a region scales; the native kernels
    each get `cores / workers` threads.

    workers: processes (0: RPPS_THREADS, else the CPU count). start_method: "fork" where there is one (cheap, and the parent's
    config paths come along), else "spawn" (a script then needs its `if __name__ == "__main__"` guard)"""
    def __init__(self, workers=0, capacity=None, min_rows=1, start_method=None):
        n = workers or int(os.environ.get("RPPS_THREADS", 0)) or os.cpu_count() or 1
        self.workers, self.min_rows = n, max(1, int(min_rows))
        self.capacity = capacity or 2*n
        self.start_method = start_method
        self._pool = None

    def prepare(self, worker):
        import multiprocessing as mp
        from .. import _config
        self.worker = worker
        method = self.start_method or ("fork" if "fork" in mp.get_all_start_methods() else "spawn")
        threads = max(1, (os.cpu_count() or 1)//self.workers)
        with warnings.catch_warnings(): # (3.12 warns about forking a process that has threads, such as a BLAS pool)
            warnings.simplefilter("ignore", DeprecationWarning)
            self._pool = mp.get_context(method).Pool(self.workers, _child_init, (worker, list(_config._EXTRA), threads))

    def submit(self, batch):
        n = min(self.workers, max(1, len(batch)//self.min_rows))
        size = -(-len(batch)//n)
        parts = [batch[i:i + size] for i in range(0, len(batch), size)]
        futs = [_Async(self._pool.apply_async(_child_run, (part,))) for part in parts]
        return futs[0] if len(futs) == 1 else _Joined(futs)

    def close(self):
        if self._pool is not None:
            self._pool.terminate()
            self._pool.join()
            self._pool = None

class _Joined:
    """The Future of a batch that ran as several parts: done when all are, its result their results in order"""
    def __init__(self, futs):
        self.futs = futs

    def done(self):
        return all(f.done() for f in self.futs)

    def result(self):
        return [r for f in self.futs for r in f.result()]

def _run(worker, batch):
    out = list(worker(batch))
    if len(out) != len(batch) or any(not isinstance(r, Result) or r.seq != f.seq for r, f in zip(out, batch)):
        raise RuntimeError("a worker gives one Result per frame, in the order given")
    return out

def make(executor="inline", **kw):
    """An executor by name: "inline", "thread" (workers=...). An Executor instance goes through"""
    if isinstance(executor, Executor):
        return executor
    kinds = {"inline": InlineExecutor, "thread": ThreadExecutor, "process": ProcessExecutor}
    if executor not in kinds:
        from ..errors import ConfigError
        raise ConfigError(f"no executor {executor!r}, have {sorted(kinds)}")
    return kinds[executor](**kw)
