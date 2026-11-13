"""The parallel runner: a serial front, then a region that runs frame by frame (README.dev.md, Parallelism)

    rx = profile.rx(parallel=rp.parallel.Parallel("thread", workers=8))
    out = rx(block)                  # as any runner; the output is the same, a frame later

The front (the steps before the profile's `parallel` marker) is an ordinary `Runner`: timing, carrier, frame sync, the
header, and the control path, which resolves the parameters of each frame. What reaches the region is cut at the frame
starts those changes mark; each frame goes out as a `Frame` with its parameters, a `Scheduler` batches them to an
`Executor`, and the results come back in order.

`PlanWorker` (rpps.parallel.worker) runs the region on a batch of frames through a `BatchPlan` of kernels, from a fresh
state per frame. A frame's output is as long as the region makes it; one that cannot be decoded is dropped and reported.

Output positions count from 0 and add up frame by frame (the serial runner starts at the first frame's position scaled by
its ratio; not done here yet), and a dropped frame gives no items and no gap. Latency is a frame, declared in
`stream.est["parallel"]`.
"""
import copy
from dataclasses import dataclass

import numpy as np

from ..base.rpps import FLUSH
from ..profile import control as _ctl
from ..profile.runner import Runner
from .frame import Frame, Result, DROPPED
from .scheduler import Scheduler
from .worker import PlanWorker

@dataclass
class Parallel:
    """How a parallel region runs: executor ("inline", "thread", or an Executor), its keyword arguments, and batching:
    frames that arrive in one call (up to max_batch, of one key) run as one batch"""
    executor: object = "inline"
    max_batch: int = 16
    kw: dict = None

    def __init__(self, executor="inline", max_batch=16, **kw):
        self.executor, self.max_batch, self.kw = executor, max_batch, kw

class ParallelRunner(Runner):
    """A Runner whose slots from `region_slot` on run frame by frame on an executor. Same contract as Runner"""
    def __init__(self, slots, name, entries, direction, make_steps, region_first, parallel):
        super().__init__(slots, name, entries, direction)
        if direction != "rx":
            raise ValueError("a parallel region is a receive-side thing")
        self.stop = next(i for i, s in enumerate(self.slots) if s.step.id == region_first)
        self.sched = Scheduler(parallel.executor, PlanWorker(make_steps, direction), parallel.max_batch, False,
                               **(parallel.kw or {}))
        self._pend, self._pend_pos, self._cur = [], None, {}
        self._fseq, self._out_next, self._tags = 0, None, None
        self.frames = 0

    def close(self):
        self.sched.close()

    def _update(self, ch):
        for i, ps in ch.params.items():
            if i >= self.stop:
                self._cur.setdefault(self.slots[i].step.id, {}).update(ps)

    def _finish(self, D):
        b, start = self.stop, D.meta.pos
        end = start + len(D)
        self._tags = (type(D), D.CAT, D.DT, D.REP, D.meta.state.Fs, D.meta.state.sps, D.meta.state.cf, D.meta.delay)
        mine = sorted(((int(round(ch.pos)), ch.seq, ch) for ch in self.changes
                       if ch.hop == b and any(i >= b for i in ch.fwd) and int(round(ch.pos)) <= end), key=lambda t: t[:2])
        results, cur = [], start
        arr = np.asarray(D.data)
        for p, _, ch in mine:
            p = min(max(p, start), end)
            self._take(arr, start, cur, p)
            results += self._close()
            self._update(ch)
            cur = max(cur, p)
            ch.fwd = {i for i in ch.fwd if i < b}
            if not ch.fwd and not ch.back:
                self.changes.remove(ch)
        self._take(arr, start, cur, end)
        results += self.sched.kick() # the frames of this call run now, as batches
        if D.flush:
            results += self._close()
            results += self.sched.flush()
        return self._emit(D, results)

    def _take(self, arr, start, a, b):
        if b > a:
            if self._pend_pos is None:
                self._pend_pos = a
            self._pend.append(arr[a - start:b - start])

    def _close(self):
        """The frame being gathered is complete: send it off. Returns the results that came back, in order"""
        if not self._pend:
            return []
        data = np.concatenate(self._pend)
        f = Frame(self._fseq, data, copy.deepcopy(self._cur), pos=self._pend_pos, tags=self._tags)
        self._pend, self._pend_pos = [], None
        self._fseq += 1
        self.frames += 1
        return self.sched.push(f)

    def _emit(self, D, results):
        meta = D.meta.fork()
        for r in results:
            for rep in r.reports:
                self._report(rep.pop("what"), **rep)
        got = [r for r in results if r.status != DROPPED]
        if got:
            self._tags_out = got[-1].tags
        cls, cat, dt, rep = getattr(self, "_tags_out", None) or (type(D), D.CAT, D.DT, D.REP)
        arrs = [r.data for r in got if len(r.data)]
        arr = np.concatenate(arrs) if arrs else np.asarray(D.data)[:0]
        meta.pos = self._out_next or 0
        self._out_next = (self._out_next or 0) + len(arr)
        ctx = dict(D.ctx)
        ctx[FLUSH] = D.flush
        self._stream.est["parallel"] = {"frames": self.frames, "in_flight": self.sched.pending, "latency": "one frame"}
        return cls(arr, cat, dt, ctx, meta, rep)
