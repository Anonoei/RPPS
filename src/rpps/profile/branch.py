"""A profile's receive side that fans out: one trunk, then branches that read the same blocks (rpps.profile)

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

With a `frame`, a branch can read only part of each frame, and only some frames:

    {"branch": "burst", "frame": 132,
     "take": {"slot_type": [[49, 54], [78, 83]], "payload": [[0, 49], [83, 132]]},   items of each frame, in order
     "select": {"payload": {"step": "fs", "when": "word == 'bs_data'"}},            frames whose fields say so
     "branches": {...}}

Frames are `frame` items each, or start wherever a trunk control drives the branch step's own `frame` (a frame sync's
`{"<branch step id>": {"frame": "frame"}}`): then each runs to the next, of any length. A range's end may be null (to
the frame's end), and `{"ranges": [...], "unit": 196}` keeps only whole groups of that many items (P25's TSBKs, one to
three a frame).

`take` gives a branch the item ranges of each frame, joined (a code split around a sync word); `select` the frames
for which an expression over the fields a trunk step published for that frame holds (the sync word that says voice or
data, `frame % 2` for one TDMA slot of two). Such a branch counts its own positions (the items it read); changes and
messages cross over at the trunk positions they stand for.

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


def take_spec(spec, where):
    """(ranges [(start, end or None)], unit) of a branch's `take`: a list of ranges, or {"ranges": [...], "unit": u}"""
    unit = 1
    if isinstance(spec, dict):
        if set(spec) - {"ranges", "unit"} or "ranges" not in spec:
            raise ConfigError(f"{where}: take is a list of [start, end] ranges, or {{\"ranges\": [...], \"unit\": items}}")
        unit, spec = int(spec.get("unit", 1)), spec["ranges"]
    ranges = []
    for r in spec:
        if not isinstance(r, (list, tuple)) or len(r) != 2:
            raise ConfigError(f"{where}: a range is [start, end] (end null: to the frame's end), not {r!r}")
        a, e = int(r[0]), None if r[1] is None else int(r[1])
        if a < 0 or (e is not None and e <= a):
            raise ConfigError(f"{where}: range {r} is empty or negative")
        ranges.append((a, e))
    if unit < 1:
        raise ConfigError(f"{where}: unit {unit}: whole groups of at least one item")
    return ranges, unit

def take_index(ranges, unit, length):
    """Items of a frame of `length` a branch reads: its ranges in order (an open end: to the frame's end), as many as
    make whole units; None when a range with an end is past the frame (the frame is not read)"""
    import numpy as np
    parts = []
    for a, e in ranges:
        if e is None:
            e = length
        elif e > length:
            return None
        parts.append(np.arange(a, max(a, e)))
    idx = np.concatenate(parts) if parts else np.zeros(0, np.int64)
    idx = idx[:len(idx)//unit*unit]
    if len(np.unique(idx)) != len(idx):
        raise ConfigError(f"take ranges {ranges} overlap")
    return idx.astype(np.int64)

class Taken:
    """Positions of a branch that reads some items of each frame (`take`) of some frames (`select`): it counts the items
    it read. Each frame read is an anchor: its first position in the branch, its start in the trunk's output, its
    length and the items read"""
    def __init__(self, ranges, unit):
        self.ranges, self.unit = ranges, unit
        self.anchors = [] # (branch position, trunk position of the frame's start, frame length, items read)
        self.count = 0    # branch items so far
        self.covered = None # trunk position up to which every frame has been looked at

    def index(self, length):
        return take_index(self.ranges, self.unit, length)

    def add(self, start, length, idx):
        """A frame looked at; idx: the items read (None: not read)"""
        if idx is not None and len(idx):
            self.anchors.append((self.count, int(start), int(length), idx))
            self.count += len(idx)
        self.covered = int(start) + int(length)

    def to_trunk(self, b):
        import bisect
        if not self.anchors:
            return b
        k = max(bisect.bisect_right([a[0] for a in self.anchors], b) - 1, 0)
        bp, tp, n, idx = self.anchors[k]
        off = int(round(b - bp))
        if off < 0:
            return tp
        return tp + int(idx[off]) if off < len(idx) else tp + n + (off - len(idx))

    def to_branch(self, t):
        """The branch position of trunk position t: the first item read at or after it"""
        import bisect
        k = bisect.bisect_right([a[1] for a in self.anchors], t) - 1
        if k < 0:
            return self.anchors[0][0] if self.anchors else self.count
        bp, tp, n, idx = self.anchors[k]
        off = int(bisect.bisect_left(idx.tolist(), t - tp))
        if off < len(idx) and t - tp < n:
            return bp + off
        return self.anchors[k + 1][0] if k + 1 < len(self.anchors) else bp + len(idx)

class BranchChain(Runner):
    """One branch's runner: it also hands what its headers publish for other branches to the bus, at trunk positions"""
    def __init__(self, slots, name, entries, branch, bus, profile=None):
        super().__init__(slots, name, entries, "rx", profile)
        self.branch, self.bus = branch, bus
        self.taken = None # Taken: positions of a branch that reads part of the frames; None: the trunk's own

    def trunk_pos(self, pos):
        return pos if self.taken is None else self.taken.to_trunk(pos)

    def branch_pos(self, pos):
        return pos if self.taken is None else self.taken.to_branch(pos)

    def _accept(self, origin, fields, at, out, span, step, group=None):
        super()._accept(origin, fields, at, out, span, step, group)
        if origin >= 0 and step in self.bus.entries:
            entry = self.bus.entries[step]
            self.bus.post(self.branch, entry, fields, self.trunk_pos(self._back(at, origin, 0)))
            if span is not None: # Fields that hold for a span go back to the defaults after it, in the driven branches too
                self.bus.post(self.branch, entry, entry.revert(fields), self.trunk_pos(self._back(at + span, origin, 0)))

    def inject_params(self, sid, params, pos, key):
        """A change from the trunk: step sid's parameters from `pos` of this branch's input"""
        if key in self._keys:
            return
        self._keys.add(key)
        pos = self.branch_pos(pos)
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
        pos = self.branch_pos(pos)
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
    kinds = None # The message kinds handed out (Profile.rx(kinds=...)), None: all

    def __init__(self, trunk, chains, name, bus, order, frame=None, exit=None, owner=None, take=None, select=None,
                 step_id=None, framed=False):
        self.trunk, self.chains, self.name, self.bus, self.order = trunk, chains, name, bus, order
        self.exit, self.owner = exit, owner or {} # The trunk's Exit (None: no control of the trunk drives a branch)
        self.frame = None if frame is None else int(frame)
        self.step_id, self.framed = step_id, bool(framed) # framed: frames start where a trunk control drives `frame`
        if self.frame is not None and self.framed:
            raise ConfigError(f"{name}: frames of a fixed `frame` length, or started by a control, not both")
        self.held = None # Trunk output short of a whole frame
        self._starts = [] # Frame starts a control gave (trunk output positions), not yet used
        self._input = (1.0, 0.0, None)
        self.take, self.select = dict(take or {}), dict(select or {})
        self._pending = [] # Trunk changes for branches that read part of the frames, until their frame has been looked at
        if (self.take or self.select) and self.frame is None and not self.framed:
            raise ConfigError(f"{name}: `take` and `select` read parts of frames: the branch step needs `frame`, or a "
                              f"control that drives its `frame`")
        for b in list(self.take) + list(self.select):
            if b not in self.chains:
                raise ConfigError(f"{name}: take / select name branch {b!r}, not one of {list(self.chains)}")
        for b in self.chains:
            if b in self.take or b in self.select:
                ranges, unit = take_spec(self.take.get(b, [[0, None]]), f"{name}: take of {b!r}")
                if self.frame is not None:
                    idx = take_index(ranges, unit, self.frame)
                    if idx is None or not len(idx):
                        raise ConfigError(f"{name}: take of {b!r}: ranges within a frame of {self.frame}")
                self.chains[b].taken = Taken(ranges, unit)
        for b, rule in self.select.items():
            if not isinstance(rule, dict) or not {"step", "when"} <= set(rule):
                raise ConfigError(f"{name}: select of {b!r} is {{\"step\": <trunk step id>, \"when\": <expression of its fields>}}")
            if trunk is None or rule["step"] not in trunk._step_slots:
                raise ConfigError(f"{name}: select of {b!r}: {rule['step']!r} is not a step of the trunk")

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
                    if sid == self.step_id: # The branch step's own `frame`: a frame starts here
                        self._starts.append(int(round(pos)))
                        continue
                    key = (EXIT, sid, int(round(pos)), tuple(sorted(params.items(), key=lambda kv: kv[0])))
                    chain = self.chains[self.owner[sid]]
                    if chain.taken is not None: # (placed once its frame has been looked at)
                        self._pending.append((chain, sid, params, pos, key))
                    else:
                        chain.inject_params(sid, params, pos, key)
        frames = []
        if self.frame is not None or self.framed:
            data, frames = self._frames(data)
        pieces = self._parts(data, frames)
        still = []
        for chain, sid, params, pos, key in self._pending:
            if chain.taken.covered is not None and pos < chain.taken.covered:
                chain.inject_params(sid, params, pos, key)
            else:
                still.append((chain, sid, params, pos, key))
        self._pending = still
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
            src = pieces.get(name, data)
            meta = src.meta.fork()
            meta.state = data.meta.state.copy()
            meta.op = data.meta.state.copy()
            meta.stream, meta.gen, meta.flush = subs[name], subs[name].gen, data.meta.flush
            piece = type(data)(src.data, data.CAT, data.DT, dict(data.ctx), meta, data.REP)
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
                at = chain.trunk_pos(m.sample)
                pos = self.trunk.to_input(at) if self.trunk is not None else at
                out.append(m.placed(pos, Fs, cf, t0))
        return sorted((m for m in out if self.kinds is None or m.kind in self.kinds), key=lambda m: m.sample)

    def _parts(self, data, frames):
        """{branch: its items of the frames [(start, end)] in data, at its own positions} for the branches that read part
        of them (`take`, `select`); every frame is looked at once, the trunk's fields deciding `select`"""
        import numpy as np
        out = {}
        parts = {name: c for name, c in self.chains.items() if c.taken is not None}
        if not parts:
            return out
        x = np.asarray(data.data)
        starts = np.array([f for f, _ in frames], dtype=np.int64)
        for name, chain in parts.items():
            t = chain.taken
            read = self._selected(name, starts, data) if name in self.select else np.ones(len(frames), bool)
            first, pieces = t.count, []
            for (f, e), r in zip(frames, read):
                idx = t.index(e - f) if r else None
                t.add(f, e - f, idx)
                if idx is not None and len(idx):
                    pieces.append(x[f - data.meta.pos + idx])
            meta = data.meta.fork()
            meta.pos = first
            sub = np.concatenate(pieces) if pieces else x[:0]
            out[name] = type(data)(sub, data.CAT, data.DT, dict(data.ctx), meta, data.REP)
        return out

    def _selected(self, name, starts, data):
        """Which frames (trunk output starts) branch `name` reads: its select expression over the fields the named trunk
        step published last at or before each frame's start, carried back to that step's input"""
        import numpy as np
        from .. import _config
        rule = self.select[name]
        sid = rule["step"]
        slot = self.trunk._step_slots[sid][0]
        ctl = data.meta.stream.control
        log = ctl.log if ctl is not None else []
        st = data.meta.stream.states.setdefault((self, "select", name), {"i": 0, "fields": {}}) # a cursor into the log
        fields = st["fields"]
        out = np.zeros(len(starts), bool)
        for i, o in enumerate(starts):
            p = self.trunk._back(o, len(self.trunk.slots), slot) # the frame's start at the publisher's input
            while st["i"] < len(log): # the step's fields published at or before it, the last of each winning
                e = log[st["i"]]
                if e["step"] == sid:
                    if e["at"] > p + 1e-6:
                        break
                    fields.update(e["fields"])
                st["i"] += 1
            try:
                out[i] = bool(fields) and bool(_config.evaluate(rule["when"], fields, f"{self.name} select {name}"))
            except ConfigError: # a field the expression needs was not published for this frame
                out[i] = False
        return out

    def _frames(self, data):
        """The whole frames of what the trunk gave so far (the end of the stream: all of it), and where each is
        [(start, end)]: every `frame` items, or from each start a control gave to the next"""
        import numpy as np
        if self.held is not None and len(self.held):
            meta = self.held.meta.fork()
            x = np.concatenate((np.asarray(self.held.data), np.asarray(data.data)))
            data = type(data)(x, data.CAT, data.DT, dict(data.ctx), meta, data.REP)
        x = np.asarray(data.data)
        pos0, end = data.meta.pos, data.meta.pos + len(x)
        if self.frame is not None:
            n = len(x)//self.frame
            frames = [(pos0 + i*self.frame, pos0 + (i + 1)*self.frame) for i in range(n)]
            cut = end if data.flush else pos0 + n*self.frame
            if data.flush and cut > pos0 + n*self.frame:
                frames.append((pos0 + n*self.frame, end))
        else:
            starts = sorted(set(p for p in self._starts if p >= pos0))
            frames = [(a, b) for a, b in zip(starts, starts[1:]) if b <= end]
            if data.flush and starts and starts[-1] < end:
                frames.append((starts[-1], end))
            cut = end if data.flush else (starts[-1] if starts else end) # (an open frame waits; before any start: none)
            self._starts = [p for p in starts if p >= cut]
        n = cut - pos0
        meta = data.meta.fork()
        meta.pos = cut
        self.held = type(data)(x[n:], data.CAT, data.DT, dict(data.ctx), meta, data.REP)
        return type(data)(x[:n], data.CAT, data.DT, dict(data.ctx), data.meta, data.REP), frames

    def __radd__(self, data):
        return self(data)
