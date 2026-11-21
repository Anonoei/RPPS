"""A profile's receive side that fans out: one trunk, then branches that read the same blocks (README.dev.md, Profiles)

    "steps": [ ...trunk steps...,
               {"branch": "channels", "order": ["pcfich", "pdcch", "pdsch"],
                "branches": {"pcfich": [...steps...], "pdcch": [...], "pdsch": [...]}}]

The branch step ends the chain. Every block the trunk gives goes through each branch, in `order`, and the runner gives
{branch name: output}. Each branch is a chain of its own (a runner, in a stream of its own, as rp.Tee's branches are), so
everything a chain does (header-driven changes at positions, re-runs, restarts) works inside one as it does in a profile.

Controls cross branches: a header in one branch may drive steps in the branches after it. Its fields' position is carried
back through its own branch to the trunk's output (the position every branch reads), and the driven branch takes the change
there, forward through its steps, before it reads the block. So a branch drives only those after it in `order` (one
before it has read the block already): the PCFICH's control format sets the PDCCH region of the same subframe, a DCI the
PDSCH's allocation.

A header publishes once it has read its frame, and the driven branch reads after it only within the block they share. So
a branch step names its `frame` (items of the trunk's output in one: an LTE subframe's resource elements): the branches read
whole frames, collected from however the blocks were cut, and every header rules on a frame before the branches after it read
that frame. Without one, the branches read each block as it comes, which suits headers whose frames never span two blocks.
"""
from ..errors import ConfigError
from ..base.meta import Stream
from .runner import Runner, Change


class BranchChain(Runner):
    """One branch's runner: it also hands what its headers publish for other branches to the bus, at trunk positions"""
    def __init__(self, slots, name, entries, branch, bus):
        super().__init__(slots, name, entries, "rx")
        self.branch, self.bus = branch, bus

    def _accept(self, origin, fields, at, out, span, step, group=None):
        super()._accept(origin, fields, at, out, span, step, group)
        if origin >= 0 and step in self.bus.entries:
            self.bus.post(self.branch, self.bus.entries[step], fields, self._back(at, origin, 0))

    def inject(self, entry, fields, pos, key):
        """A change from another branch's header, at `pos` of this branch's input (a trunk position)"""
        if key in self._keys:
            return
        self._keys.add(key)
        values, _ = entry.values(fields)
        params = {i: dict(ps) for sid, ps in values.items() for i in self._step_slots.get(sid, [])}
        if not params:
            return
        ch = Change()
        ch.key, ch.fields, ch.origin, ch.at, ch.params, ch.step = key, dict(fields), -1, pos, params, entry.header
        self._seq += 1
        ch.seq = self._seq
        ch.fwd, ch.hop, ch.pos = set(params), 0, pos
        ch.group, ch.btargets, ch.back, ch.s0 = (-1, pos), [], {}, None
        self.changes.append(ch)


class _Bus:
    """Publications on their way from a branch's header to the branches after it"""
    def __init__(self, entries, owner):
        self.entries, self.owner = entries, owner # owner: step id -> branch name
        self.queue = []

    def post(self, branch, entry, fields, pos):
        targets = {self.owner[sid] for sid in entry.drive if sid in self.owner} - {branch}
        if targets:
            key = (entry.header, int(round(pos)), tuple(sorted(fields.items(), key=lambda kv: kv[0])))
            self.queue.append((targets, entry, dict(fields), pos, key, branch))


class BranchRunner:
    """trunk: the runner of the steps before the branch step (None: none), chains: {name: BranchChain} in order"""
    def __init__(self, trunk, chains, name, bus, order, frame=None):
        self.trunk, self.chains, self.name, self.bus, self.order = trunk, chains, name, bus, order
        self.frame = None if frame is None else int(frame)
        self.held = None # Trunk output short of a whole frame

    def __repr__(self):
        return f"<BranchRunner {self.name}: {self.trunk!r} | " + ", ".join(f"{k}: {v!r}" for k, v in self.chains.items()) + ">"

    @property
    def ops(self):
        return list(self.trunk.ops) if self.trunk is not None else []

    def __call__(self, data, fields=None):
        if fields is not None:
            raise ConfigError(f"{self.name}: fields are for transmitting")
        if self.trunk is not None:
            data = self.trunk(data)
        if self.frame is not None:
            data = self._frames(data)
        parent = data.meta.stream
        subs = parent.states.get(self)
        if subs is None: # The branches' streams, made when the runner first runs in this stream (as rp.Tee's)
            subs = parent.states[self] = {k: Stream(parent.src_Fs) for k in self.chains}
            for s in subs.values():
                s.t0 = parent.t0
        out = {}
        for name in self.order:
            for targets, entry, fields, pos, key, frm in self.bus.queue:
                if name in targets: # (a branch drives only those after it: the profile checks)
                    self.chains[name].inject(entry, fields, pos, key)
            meta = data.meta.fork()
            meta.state = data.meta.state.copy()
            meta.op = data.meta.state.copy()
            meta.stream, meta.gen, meta.flush = subs[name], subs[name].gen, data.meta.flush
            piece = type(data)(data.data, data.CAT, data.DT, dict(data.ctx), meta, data.REP)
            out[name] = self.chains[name](piece)
        self.bus.queue = [q for q in self.bus.queue if q[3] >= data.meta.pos + len(data)] # (kept: positions ahead)
        return out

    def _frames(self, data):
        """The whole frames of what the trunk gave so far (the end of the stream: all of it)"""
        import numpy as np
        if self.held is not None and len(self.held):
            meta = self.held.meta.fork()
            x = np.concatenate((np.asarray(self.held.data), np.asarray(data.data)))
            data = type(data)(x, data.CAT, data.DT, dict(data.ctx), meta, data.REP)
        n = len(data) if data.flush else len(data)//self.frame*self.frame
        x = np.asarray(data.data)
        meta = data.meta.fork()
        meta.pos = data.meta.pos + n
        self.held = type(data)(x[n:], data.CAT, data.DT, dict(data.ctx), meta, data.REP)
        return type(data)(x[:n], data.CAT, data.DT, dict(data.ctx), data.meta, data.REP)

    def __radd__(self, data):
        return self(data)
