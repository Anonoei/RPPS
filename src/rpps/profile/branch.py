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

A header in the trunk drives steps in the branches too (a DVB-S2 PLS sets the BBFRAME length of every input stream's
BBHEADER). The profile ends the trunk on an exit step, `EXIT`, that the controls drive in their place: the runner carries
the change forward through the trunk with the data as it does any, and applies it to the exit at the trunk's output
position, which is where every branch takes it.
"""
from ..errors import ConfigError
from ..base import Stage
from ..base.meta import Stream
from .runner import Runner, Change

EXIT = "@branches" # The trunk's exit step: what the trunk's controls drive in the branches' place


class Exit(Stage):
    """The trunk's last step: passes the data through, and keeps every change applied to it with its position (the trunk's
    output position from which it holds). Its parameters are "<step id>/<parameter>" of steps in the branches"""
    def __init__(self, params):
        self.CONTROLS = {p: "live" for p in params}
        self.records, self.cursor, self._pending = [], None, []

    def __str__(self):
        return "Exit"

    def takes(self, encode=True):
        return None # (anything the trunk gives)

    def control(self, **params):
        by_step = {}
        for k, v in params.items():
            sid, _, p = k.partition("/")
            by_step.setdefault(sid, {})[p] = v
        if self.cursor is None: # Before any data: from where the first block starts
            self._pending.append(by_step)
        else:
            self.records.append((self.cursor, by_step))

    def step(self, x, st, meta):
        if self._pending:
            self.records += [(meta.pos, b) for b in self._pending]
            self._pending = []
        self.cursor = meta.pos + len(x)
        return x

    def take(self):
        out, self.records = self.records, []
        return out


class BranchChain(Runner):
    """One branch's runner: it also hands what its headers publish for other branches to the bus, at trunk positions"""
    def __init__(self, slots, name, entries, branch, bus, profile=None):
        super().__init__(slots, name, entries, "rx", profile)
        self.branch, self.bus = branch, bus

    def _accept(self, origin, fields, at, out, span, step, group=None):
        super()._accept(origin, fields, at, out, span, step, group)
        if origin >= 0 and step in self.bus.entries:
            entry = self.bus.entries[step]
            self.bus.post(self.branch, entry, fields, self._back(at, origin, 0))
            if span is not None: # Fields that hold for a span go back to the defaults after it, in the driven branches too
                self.bus.post(self.branch, entry, entry.revert(fields), self._back(at + span, origin, 0))

    def inject_params(self, sid, params, pos, key):
        """A change from the trunk: step sid's parameters from `pos` of this branch's input"""
        if key in self._keys:
            return
        self._keys.add(key)
        slots = {i: dict(params) for i in self._step_slots.get(sid, [])}
        if not slots:
            return
        ch = Change()
        ch.key, ch.fields, ch.origin, ch.at, ch.params, ch.step = key, dict(params), -1, pos, slots, EXIT
        self._seq += 1
        ch.seq = self._seq
        ch.fwd, ch.hop, ch.pos = set(slots), 0, pos
        ch.group, ch.btargets, ch.back, ch.s0 = (-1, pos), [], {}, None
        self.changes.append(ch)

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
    def __init__(self, trunk, chains, name, bus, order, frame=None, exit=None, owner=None):
        self.trunk, self.chains, self.name, self.bus, self.order = trunk, chains, name, bus, order
        self.exit, self.owner = exit, owner or {} # The trunk's Exit (None: no control of the trunk drives a branch)
        self.frame = None if frame is None else int(frame)
        self.held = None # Trunk output short of a whole frame
        self._input = (1.0, 0.0, None)

    def __repr__(self):
        return f"<BranchRunner {self.name}: {self.trunk!r} | " + ", ".join(f"{k}: {v!r}" for k, v in self.chains.items()) + ">"

    @property
    def ops(self):
        return list(self.trunk.ops) if self.trunk is not None else []

    def __call__(self, data, fields=None):
        if fields is not None:
            raise ConfigError(f"{self.name}: fields are for transmitting")
        self._input = (data.meta.Fs or 1.0, data.meta.cf, data.meta.stream.t0)
        if self.trunk is not None:
            data = self.trunk(data)
        if self.exit is not None:
            for pos, by_step in self.exit.take():
                for sid, params in by_step.items():
                    key = (EXIT, sid, int(round(pos)), tuple(sorted(params.items(), key=lambda kv: kv[0])))
                    self.chains[self.owner[sid]].inject_params(sid, params, pos, key)
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

    def take_messages(self):
        """The frames the trunk and every branch decoded since the last call, placed in the stream given (a branch's
        positions are the trunk's output, carried back through it)"""
        Fs, cf, t0 = self._input
        out = list(self.trunk.take_messages()) if self.trunk is not None else []
        for chain in self.chains.values():
            for m in chain.take_messages():
                pos = self.trunk.to_input(m.sample) if self.trunk is not None else m.sample
                out.append(m.placed(pos, Fs, cf, t0))
        return sorted(out, key=lambda m: m.sample)

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
