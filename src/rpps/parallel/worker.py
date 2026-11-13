"""The worker of a parallel region: batches of frames through a plan (README.dev.md, Parallelism)

    worker(frames) -> [Result]       frames share a key; one plan per key, compiled on first use and kept

The region's operators are built once per worker (building takes a moment) and shared by its plans: a `LoopKernel` applies its
parameters to every frame as a frame start, so frames of different keys can follow one another. A worker is used by one
thread at a time (`per_thread` makes another).
"""
import numpy as np

from .frame import Frame, Form, Result
from .kernel import Batch
from .plan import compile

class RegionSteps:
    """The built steps of a profile's parallel region, as a picklable recipe (a worker in another process builds its own)"""
    def __init__(self, name, obj, sync, params, first):
        self.name, self.obj, self.sync, self.params, self.first = name, obj, sync, params, first

    def __call__(self):
        from ..profile.profile import Profile
        return Profile(self.name, self.obj, sync=self.sync, params=self.params)._build()[self.first:]

class PlanWorker:
    def __init__(self, make_steps, direction="rx", xp=np):
        self.make_steps, self.direction, self.xp = make_steps, direction, xp
        self.steps = list(make_steps())
        self.plans = {}

    def __reduce__(self):
        # What crosses to another process is the recipe: the operators and plans are built there
        return (PlanWorker, (self.make_steps, self.direction))

    def per_thread(self):
        return PlanWorker(self.make_steps, self.direction, self.xp)

    def plan(self, frame):
        key = frame.key
        plan = self.plans.get(key)
        if plan is None:
            plan = self.plans[key] = compile(self.steps, frame.params, self.direction, key)
        return plan

    def __call__(self, frames):
        if not frames:
            return []
        keys = {f.key for f in frames}
        if len(keys) != 1:
            raise ValueError(f"a batch holds frames of one key, got {len(keys)}")
        plan = self.plan(frames[0])
        form = Form(*frames[0].tags)
        b = Batch(np.stack([f.data for f in frames]), form, [f.pos for f in frames], len(frames[0].data))
        b = plan.run(self.xp, b)
        out = []
        for i, f in enumerate(frames):
            if b.bad[i]:
                r = Result.dropped(f, b.why.get(i, "failed"))
            else:
                y = np.asarray(b.x[i])
                r = Result(f.seq, y, f.pos_out, len(y), reports=list(b.notes.get(i, [])), tags=tuple(b.form[:4]))
            out.append(r)
        return out
