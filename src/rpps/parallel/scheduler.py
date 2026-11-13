"""Scheduler: frames in, batches out to an executor, results back in order (README.dev.md, Parallelism)

    sch = Scheduler(executor, worker, max_batch=16)
    for frame in frames:
        for result in sch.push(frame):       # whatever is ready, in `seq` order
            ...
    for result in sch.flush():               # the end of the stream: everything still in flight
        ...

Frames batch by `Frame.key` (same parameters and shape), up to `max_batch`; see `eager`. At most `executor.capacity` batches are in flight:
a push that would exceed it waits for the oldest, so a slow back stalls the front instead of growing memory. Results are
released in `seq` order only, so a frame that finishes early waits for those before it; a gap (a dropped frame) is a Result
with status DROPPED, which keeps its place and its output position.
"""
from collections import OrderedDict, deque

from .executor import make

class Scheduler:
    def __init__(self, executor="inline", worker=None, max_batch=16, eager=True, **kw):
        self.executor = make(executor, **kw)
        self.executor.prepare(worker)
        self.max_batch, self.eager = max(1, int(max_batch)), eager
        self._open = OrderedDict() # key -> frames waiting for a batch
        self._flying = deque()     # (batch, future), oldest first
        self._ready = {}           # seq -> Result, finished but not yet in order
        self._next = 0             # the seq to release next
        self._seen = 0             # frames pushed
        self.max_flying = 0

    @property
    def pending(self):
        """Frames pushed and not yet released"""
        return self._seen - self._next

    def push(self, frame):
        """Take a frame (they come in `seq` order); give back the results that are ready, in order"""
        if frame.seq != self._seen:
            raise ValueError(f"frames come in order: expected seq {self._seen}, got {frame.seq}")
        self._seen += 1
        q = self._open.setdefault(frame.key, [])
        q.append(frame)
        if len(q) >= self.max_batch:
            self._submit(self._open.pop(frame.key))
        # Idle capacity is not left waiting for a full batch: batches only grow while every slot is busy, so a trickle of
        # frames is not held back (`eager=False` waits for max_batch or the flush instead, for a device that wants full batches)
        while self.eager and self._open and len(self._flying) < self.executor.capacity:
            self._submit(self._open.pop(next(iter(self._open))))
        return self._release()

    def kick(self):
        """Run what is waiting now, whatever its size (the frames that are here, all pushed): a caller that pushes a group of
        frames and then waits for results calls this, so batches form from the group without waiting for `max_batch`.
        Gives back the results that are ready, in order"""
        for key in list(self._open):
            self._submit(self._open.pop(key))
        return self._release()

    def flush(self):
        """The stream ends: run what is waiting and give every result left, in order"""
        for key in list(self._open):
            self._submit(self._open.pop(key))
        while self._flying:
            self._wait()
        return self._release()

    def _submit(self, batch):
        while len(self._flying) >= self.executor.capacity:
            self._wait()
        self._flying.append((batch, self.executor.submit(batch)))
        self.max_flying = max(self.max_flying, len(self._flying))

    def _wait(self):
        batch, fut = self._flying.popleft()
        for r in fut.result():
            self._ready[r.seq] = r

    def _release(self):
        # A batch can finish out of order, so look at all of them
        for item in list(self._flying):
            if item[1].done():
                self._flying.remove(item)
                for r in item[1].result():
                    self._ready[r.seq] = r
        out = []
        while self._next in self._ready:
            out.append(self._ready.pop(self._next))
            self._next += 1
        return out

    def close(self):
        self.executor.close()

    def __enter__(self):
        return self
    def __exit__(self, *exc):
        self.close()
