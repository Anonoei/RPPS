"""Profile runners: a profile's operators on a stream of blocks, with header-driven control (ARCHITECTURE.md)

A runner holds one slot per operator. Each slot knows its step, how far it has read its input, and how its input
positions map to its output positions (control.PosMap). Changes come from the control path (rpps.profile.control):

  * ahead: the change is for a step after the one that published it (a DVB-S2 PLS sets the constellation of its own
    payload, which has not been demapped yet). It travels with the data like a tag: before a slot runs, the runner splits
    the slot's input at the change's position and applies it between the two parts; after the slot ran, the position is
    carried to its output through the slot's map, and on to the next target.
  * late: the change is for a step before the one that published it (an 802.11b PLCP header sets the rate of the payload
    the demapper already demapped as DBPSK). Its position is carried back through the maps to that step, and the runner
    re-runs from the last checkpoint before it: the steps from the earliest such target (the re-run span) run piece by
    piece, with a checkpoint of their state at the start of every piece and the span's input kept back, sized from the
    signal (`reach` in the controls, carried into the span's units through the current rates). A change older than the
    oldest checkpoint is applied from there instead, and the drop is reported on the stream.

Transmitting, the runner itself publishes each frame's fields at the first step: the caller's, else the profile's
schedule, else the defaults. Those drive the steps up to the header step; a header step that publishes in turn (an
802.11b PLCP builder, which knows where its payload starts) drives the steps after it.
"""
import copy

import numpy as np

from ..base.rpps import FLUSH
from . import control as _ctl
from .steps import STAGE, DECODE

MIN_SEGMENT = 1024 # Items at the re-run span's first step per piece, at least
MAX_REWINDS = 256  # Re-runs in one call before late changes are only reported

class Slot:
    """One operator of a runner and its stream positions"""
    __slots__ = ("how", "op", "step", "index", "map", "in_end", "out_next", "params", "rate")
    def __init__(self, how, op, step, index):
        self.how, self.op, self.step, self.index = how, op, step, index
        self.map = _ctl.PosMap()
        self.in_end = None   # Input position read up to
        self.out_next = None # Pipes: the position of their next output item (stages keep theirs in the stream)
        self.params = {}     # Controlled parameters, as last applied
        self.rate = None     # Output per input item, measured, for an operator that can't say

    def __repr__(self):
        return f"<Slot {self.index} {self.how}{self.op} ({self.step.id})>"

class Change:
    """Field values on their way to the steps they drive

    params: {slot: {parameter: value}}. Ahead targets are applied as the tag (`hop`, `pos`: a position in slot hop's input)
    reaches them; late targets have their position in `back` ({slot: input position}); `s0` is the position at the start
    of the re-run span"""
    __slots__ = ("key", "fields", "origin", "at", "seq", "params", "hop", "pos", "fwd", "back", "s0", "step", "group", "btargets")

    def copy(self):
        c = Change.__new__(Change)
        for k in Change.__slots__:
            setattr(c, k, getattr(self, k))
        c.fwd, c.back = set(self.fwd), dict(self.back)
        return c

    def __repr__(self):
        return f"<Change {self.step} {self.fields} at {self.at}: fwd {sorted(self.fwd)} @{self.hop}:{self.pos} back {self.back}>"

class _Ckpt:
    __slots__ = ("pos", "ctx", "slots", "states", "out_pos", "changes", "keys", "call", "outs", "est", "log")

def _attrs(op):
    """An operator's attributes, to put back after a re-run (its parameters: what `control` changes)"""
    d = dict(vars(op)) if hasattr(op, "__dict__") else {}
    for cls in type(op).__mro__:
        for k in getattr(cls, "__slots__", ()):
            if k not in ("__dict__", "__weakref__") and hasattr(op, k):
                d.setdefault(k, getattr(op, k))
    return d

def _set_attrs(op, d):
    for k, v in d.items():
        setattr(op, k, v)

class Runner:
    """Operators in order, run on a stream of blocks: `out = runner(block)`; transmitting, `runner(block, fields=...)`

    Stages keep their state in the block's stream; encoders and decoders keep theirs in the data's ctx, which the runner
    carries from one call to the next. So call one runner with consecutive blocks of one stream, and the output is the same
    whatever the cut. A new stream wants a new runner (`profile.rx()` / `profile.tx()`); each has its own operators.

    fields (transmit): the fields of the frames that start in this block, over the profile's schedule: one dict for all of
    them, or a list, one per frame in order (then the schedule)
    """
    def __init__(self, slots, name="", entries=None, direction="rx"):
        self.slots, self.name, self.direction = list(slots), name, direction
        self.ctx = None
        self.entries = dict(entries or {})
        self._step_slots = {}
        for s in self.slots:
            self._step_slots.setdefault(s.step.id, []).append(s.index)
        self.entries = {h: e for h, e in self.entries.items() if h in self._step_slots}
        self.changes, self._keys, self._logent, self._seq, self._call = [], set(), {}, 0, 0
        self._fresh = []
        self.emitted = None
        self._stream = None
        # Transmit: the entry whose fields the runner publishes, frame by frame
        self.framer = None
        if direction == "tx":
            framers = [e for e in self.entries.values() if e.schedule or e.frame is not None] or list(self.entries.values())
            if len(framers) > 1:
                from ..errors import ConfigError
                raise ConfigError(f"{name}: several controls could frame the transmit side: {[e.header for e in framers]}")
            self.framer = framers[0] if framers else None
        self.next_frame, self._sched, self._queue = None, 0, []
        # Receive: the re-run span, from the earliest step a header drives that comes before the header
        self.span = None
        if direction == "rx":
            for e in self.entries.values():
                h = self._header_slot(e)
                for sid in e.drive:
                    for i in self._step_slots.get(sid, []):
                        if i < h:
                            self.span = i if self.span is None else min(self.span, i)
        self.ckpts, self.buf = [], []
        self._span_ops = set()

    # ---- the operators

    @property
    def ops(self):
        return [(s.how, s.op) for s in self.slots]

    def __iter__(self):
        return iter(self.ops)

    def __len__(self):
        return len(self.slots)

    def __repr__(self):
        return f"<Runner {self.name}: " + " ".join(f"{how}{op}" for how, op in self.ops) + ">"

    def _header_slot(self, entry):
        return self._step_slots[entry.header][-1]

    # ---- one call

    def __call__(self, data, fields=None):
        self._call += 1
        self._stream = data.meta.stream
        _ctl.of(self._stream)
        if self.ctx is not None:
            ctx = dict(self.ctx)
            ctx[FLUSH] = data.ctx.get(FLUSH, False)
            data = data.derive(data.data, data.CAT, data.DT, ctx=ctx, rep=data.REP)
        if self.framer is not None:
            self._frames(data, fields)
        s0 = self.span if self.span is not None else len(self.slots)
        for i in range(s0):
            data = self._run_slot(i, data)
        if s0 < len(self.slots):
            data = self._run_span(data)
        self.ctx = dict(data.ctx)
        return data

    def _run_slot(self, i, data):
        """Slot i on a block: split where changes for it fall, apply them in between, then carry the others' positions on"""
        slot = self.slots[i]
        start = data.meta.pos
        end = start + len(data)
        if slot.in_end is None:
            slot.in_end = start
        todo = []
        for ch in self.changes:
            if i in ch.back:
                todo.append((ch.back[i], ch.seq, ch))
            elif ch.hop == i and i in ch.fwd:
                todo.append((ch.pos, ch.seq, ch))
        todo = sorted((int(round(p)), s, ch) for p, s, ch in todo if int(round(p)) < end)
        cuts = sorted({start, end} | {p for p, _, _ in todo if p > start})
        segments = list(zip(cuts[:-1], cuts[1:])) or [(start, end)]
        flush = data.flush
        ctx = dict(data.ctx)
        outs = []
        k = 0
        for a, b in segments:
            while k < len(todo) and todo[k][0] <= a:
                p, _, ch = todo[k]
                if p < a and p < slot.in_end:
                    self._report("late", step=slot.step.id, fields=ch.fields, at=p, applied=a)
                self._apply(slot, ch.params[i], a)
                if i in ch.back:
                    del ch.back[i]
                else:
                    ch.fwd.discard(i)
                k += 1
            piece = data if len(segments) == 1 else self._slice(data, a - start, b - start, ctx, flush and b == end)
            out = self._run_op(slot, piece)
            ctx = dict(out.ctx)
            outs.append(out)
        out = self._join(outs)
        slot.in_end = max(slot.in_end, end)
        # Changes waiting at this slot's input go on to its output once the slot has read past them
        for ch in self.changes:
            if ch.hop == i and i not in ch.fwd and ch.fwd and ch.pos < slot.in_end:
                ch.pos, ch.hop = slot.map.fwd(ch.pos), i + 1
        self.changes = [ch for ch in self.changes if ch.fwd or ch.back]
        return out

    def _run_op(self, slot, piece):
        how, op = slot.how, slot.op
        out = piece + op if how == STAGE else piece / op if how == DECODE else piece * op
        rate = _ctl.rate_of(op, how)
        if how == STAGE:
            if rate is None and piece.meta.Fs and out.meta.Fs:
                rate = out.meta.Fs/piece.meta.Fs
        else:
            if rate is None:
                if slot.rate is None and len(piece) and len(out):
                    slot.rate = len(out)/len(piece)
                rate = slot.rate
            m = out.meta.fork()
            if slot.out_next is None:
                slot.out_next = int(round(piece.meta.pos*(rate or 1.0)))
            m.pos = slot.out_next
            slot.out_next += len(out)
            out.meta = m
        if not slot.map:
            slot.map.add(piece.meta.pos, out.meta.pos, rate or 1.0)
        ctl = self._stream.control
        if ctl is not None and (ctl.anchors or ctl.published):
            mine = [a for a in ctl.anchors if a[0] is op]
            ctl.anchors = [a for a in ctl.anchors if a[0] is not op]
            for _, at, o, r in mine:
                slot.map.add(at, out.meta.pos + o, r if r is not None else (rate or slot.map.rate_at(at)))
            pubs = [p for p in ctl.published if p[0] is op]
            ctl.published = [p for p in ctl.published if p[0] is not op]
            for _, fields, at, o, span, group in pubs:
                self._accept(slot.index, fields, at, None if o is None else out.meta.pos + o, span, slot.step.id, group)
        return out

    @staticmethod
    def _slice(data, a, b, ctx, flush):
        """Items a..b of a block, as a block of the same stream at its own position"""
        meta = data.meta.fork()
        meta.pos = data.meta.pos + a
        c = dict(ctx)
        c[FLUSH] = bool(flush)
        return type(data)(np.asarray(data.data)[a:b], data.CAT, data.DT, c, meta, data.REP)

    @staticmethod
    def _join(outs):
        if len(outs) == 1:
            return outs[0]
        last = outs[-1]
        arrs = [np.asarray(o.data) for o in outs if len(o)]
        arr = np.concatenate(arrs) if arrs else np.asarray(last.data)
        meta = last.meta.fork()
        meta.pos = outs[0].meta.pos
        return type(last)(arr, last.CAT, last.DT, dict(last.ctx), meta, last.REP)

    # ---- changes

    def _apply(self, slot, params, pos):
        """Apply controlled parameters to a slot from input position `pos`: in place (live) or by rebuilding (restart)"""
        st = slot.step
        live = {p: v for p, v in params.items() if st.controls.get(p) == _ctl.LIVE}
        restart = {p: v for p, v in params.items() if st.controls.get(p) == _ctl.RESTART}
        if restart:
            new = st.rebuild({**slot.params, **restart})
            ops = [op for _, op in (new.rx if self.direction == "rx" else new.tx)]
            mine = [s for s in self.slots if s.step is st]
            ops_old = [s.op for s in mine]
            if len(ops) != len(mine):
                raise RuntimeError(f"{st.id}: a restart rebuilt {len(ops)} operators for {len(mine)}")
            for s, op in zip(mine, ops):
                s.op = op
            self._report("restart", step=st.id, params=restart, at=pos, discarded=[str(o) for o in ops_old])
        if live and hasattr(slot.op, "control"):
            slot.op.control(**live)
        slot.params.update(params)
        rate = _ctl.rate_of(slot.op, slot.how)
        if rate is None:
            rate = slot.map.rate_at(pos)
        slot.map.add(pos, slot.map.fwd(pos), rate)

    def _accept(self, origin, fields, at, out, span, step, group=None):
        """A publication: log it, and make the changes it drives (and their end, after `span` input items)"""
        key = (origin, int(at), tuple(sorted(fields.items(), key=lambda kv: kv[0])))
        if key in self._keys:
            return
        self._keys.add(key)
        entry_log = {"step": step, "fields": dict(fields), "at": int(at)}
        self._logent[key] = (origin, entry_log)
        _ctl.of(self._stream).log.append(entry_log)
        entry = self.entries.get(step) if origin >= 0 else self.framer
        if entry is None:
            return
        group = (origin, at if group is None else group)
        made = [self._make(entry, origin, fields, at, out, key, group)]
        if span is not None:
            r = self.slots[origin].map.rate_at(at) if origin >= 0 else 1.0
            made.append(self._make(entry, origin, entry.revert(fields), at + span,
                                   None if out is None else out + span*r, key + ("until",), group))
        made = [c for c in made if c is not None]
        if self.span is not None and origin >= self.span:
            self._fresh += made # Placed once the piece is through (see _settle)
        else:
            for ch in made:
                self._place(ch)

    def _make(self, entry, origin, fields, at, out, key, group):
        values, row = entry.values(fields)
        if entry.table is not None and row is None and all(fields.get(k) is not None for k in entry.table.key):
            self._report("no row", step=entry.header, fields=dict(fields), table=entry.table.name, at=at)
        h = self._header_slot(entry)
        params = {}
        for sid, ps in values.items():
            for i in self._step_slots.get(sid, []):
                if i == origin or (origin >= 0 and self.slots[i].step is self.slots[origin].step):
                    continue # A step never drives itself
                if self.direction == "tx" and ((origin < 0 and i > h) or (origin >= 0 and i < origin)):
                    continue # Transmitting, the runner drives up to the header, the header what follows
                params[i] = dict(ps)
        if not params:
            return None
        ch = Change()
        ch.key, ch.fields, ch.origin, ch.at, ch.params, ch.step = key, dict(fields), origin, at, params, entry.header
        self._seq += 1
        ch.seq = self._seq
        ch.fwd = {i for i in params if i > origin}
        ch.hop, ch.pos = origin + 1, (at if origin < 0 else out)
        if ch.fwd and ch.pos is None:
            ch.fwd = set() # The publisher said none of its output is affected
        ch.group = group
        ch.btargets = sorted((i for i in params if i < origin), reverse=True)
        ch.back, ch.s0 = {}, None
        if not ch.fwd and not ch.btargets:
            return None
        self.changes.append(ch)
        return ch

    def _place(self, ch, cut=False):
        """Carry a change's position back to the steps before its publisher that it drives. cut: forget what those steps'
        maps say after it (the pass that made them is being re-run with the change), so a later position of the same header
        (the end of its span) carries back through what the change will make of them"""
        if not ch.btargets:
            return
        p, low = ch.at, min(ch.btargets + ([self.span] if self.span is not None else []))
        for j in range(ch.origin - 1, low - 1, -1):
            slot = self.slots[j]
            p = slot.map.back(p)
            if cut:
                slot.map.cut(p)
            if j in ch.params:
                if self.span is None or j < self.span:
                    if p < (slot.in_end or 0):
                        self._report("late", step=slot.step.id, fields=dict(ch.fields), at=p, note="no re-run buffer")
                ch.back[j] = p
                self._provisional(slot, ch.params[j], p)
            if j == self.span:
                ch.s0 = p

    def _settle(self):
        """The changes published in the piece just run: placed, and if any came too late for a step, the group of the
        earliest (the publications of one header), placed again with the maps cut at each. Returns that group, or None.
        Late changes of later groups are forgotten: they came from data the re-run will process again"""
        fresh, self._fresh = self._fresh, []
        for ch in fresh:
            self._place(ch)
        late = [ch for ch in fresh if any(p < (self.slots[i].in_end or 0) for i, p in ch.back.items())]
        if not late:
            return None
        first = min(late, key=lambda c: (c.s0 if c.s0 is not None else min(c.back.values()), c.seq)).group
        sel = sorted((c for c in fresh if c.group == first), key=lambda c: (c.at, c.seq))
        for ch in sel:
            ch.back, ch.s0 = {}, None
            self._place(ch, cut=True)
        drop = {id(c) for c in fresh if c.group != first and c in late}
        self.changes = [c for c in self.changes if id(c) not in drop]
        return sel

    def _back(self, pos, frm, to):
        """A position in slot frm's input, in slot to's input (to <= frm), through the maps of the slots between"""
        p = pos
        for j in range(frm - 1, to - 1, -1):
            p = self.slots[j].map.back(p)
        return p

    def _provisional(self, slot, params, pos):
        """A late change's rate, put in the slot's map now, so later positions carry back through it right"""
        live = {p: v for p, v in params.items() if slot.step.controls.get(p) == _ctl.LIVE}
        if not live or not hasattr(slot.op, "control"):
            return
        tmp = copy.copy(slot.op)
        tmp.control(**live)
        r = _ctl.rate_of(tmp, slot.how)
        if r is not None and r != slot.map.rate_at(pos):
            slot.map.add(int(round(pos)), slot.map.fwd(pos), r)

    def _report(self, what, **kw):
        if self._stream is not None:
            _ctl.of(self._stream).reports.append({"what": what, **kw})

    # ---- transmit: the fields of each frame

    def _frames(self, data, fields):
        entry = self.framer
        if isinstance(fields, (list, tuple)):
            self._queue += [dict(f) for f in fields]
            fields = None
        start, end = data.meta.pos, data.meta.pos + len(data)
        if self.next_frame is None:
            self.next_frame = start
        if entry.frame is None: # One frame per call
            if len(data):
                self._accept(-1, self._next_fields(entry, fields), start, start, None, entry.header)
                self.next_frame = end
            return
        while self.next_frame < end:
            f = self._next_fields(entry, fields)
            self._accept(-1, f, self.next_frame, self.next_frame, None, entry.header)
            self.next_frame += entry.frame_items(f)

    def _next_fields(self, entry, fields):
        f = {k: v for k, v in entry.defaults.items() if v is not None}
        if entry.schedule:
            f.update(entry.schedule[self._sched % len(entry.schedule)])
            self._sched += 1
        if self._queue:
            f.update(self._queue.pop(0))
        elif fields:
            f.update(fields)
        return f

    # ---- receive: the re-run span

    def _span_slots(self):
        return self.slots[self.span:]

    def _owned(self, key):
        return isinstance(key, tuple) and len(key) and id(key[0]) in self._span_ops

    def _segment(self):
        """Items per piece at the span's first step, and how far back checkpoints are kept: from `reach` (header input
        items a field can apply before where it is decoded), carried into the span's units through the current rates"""
        reach = 0.0
        for e in self.entries.values():
            h = self._header_slot(e)
            if not any(i < h for sid in e.drive for i in self._step_slots.get(sid, [])):
                continue
            r = float(e.reach or 256)
            for j in range(h - 1, self.span - 1, -1):
                r /= self.slots[j].map.rate_at(self.slots[j].in_end or 0) or 1.0
            reach = max(reach, r)
        seg = max(MIN_SEGMENT, int(reach) + 1)
        return seg, reach + seg

    def _checkpoint(self, pos, ctx, outs):
        if self.ckpts and self.ckpts[-1].pos == pos:
            self.ckpts.pop()
        c = _Ckpt()
        c.pos, c.call, c.outs = pos, self._call, outs
        c.ctx = {k: copy.deepcopy(v) for k, v in ctx.items()}
        c.slots = [(s, s.op, _attrs(s.op), s.map.copy(), s.in_end, s.out_next, dict(s.params), s.rate) for s in self._span_slots()]
        st = self._stream
        c.states = {id(s.op): (s.op, copy.deepcopy(st.states.get(s.op))) for s in self._span_slots() if s.op in st.states}
        c.out_pos = {id(s.op): (s.op, st.out_pos.get(s.op)) for s in self._span_slots()}
        c.changes = [ch.copy() for ch in self.changes]
        c.keys = set(self._keys)
        c.est = copy.deepcopy(st.est)
        c.log = len(_ctl.of(st).log)
        self.ckpts.append(c)

    def _restore(self, c):
        st = self._stream
        for s, op, attrs, m, in_end, out_next, params, rate in c.slots:
            s.op = op
            _set_attrs(op, attrs)
            s.map, s.in_end, s.out_next, s.params, s.rate = m.copy(), in_end, out_next, dict(params), rate
        for s in self._span_slots():
            st.states.pop(s.op, None)
            st.out_pos.pop(s.op, None)
        for op, state in c.states.values():
            st.states[op] = copy.deepcopy(state)
        for op, pos in c.out_pos.values():
            if pos is not None:
                st.out_pos[op] = pos
        self.changes = [ch.copy() for ch in c.changes]
        self._keys = set(c.keys)
        st.est.clear()
        st.est.update(copy.deepcopy(c.est))
        # What the span published after the checkpoint is published again by the re-run (or was wrong)
        ctl = _ctl.of(st)
        span_ids = {id(e) for o, e in self._logent.values() if o >= self.span}
        ctl.log = ctl.log[:c.log] + [e for e in ctl.log[c.log:] if id(e) not in span_ids]

    def _rewind(self, late):
        """Back to the last checkpoint before the earliest late change; None when re-running is given up"""
        ctl = _ctl.of(self._stream)
        p = min(ch.s0 if ch.s0 is not None else min(ch.back.values()) for ch in late)
        fit = [c for c in self.ckpts if c.pos <= p]
        if fit:
            c = fit[-1]
        else:
            c = self.ckpts[0]
            for ch in late:
                lost = c.pos - (ch.s0 if ch.s0 is not None else c.pos)
                self._report("dropped", step=ch.step, fields=dict(ch.fields), at=ch.at, applied=c.pos, lost=lost,
                             note="the change came after its data left the re-run buffer")
        self.ckpts = self.ckpts[:self.ckpts.index(c) + 1] # What came after it is being re-run
        self._restore(c)
        for k in dict.fromkeys(ch.key[:3] for ch in late): # The header that caused it stays in the log
            e = self._logent.get(k)
            if e is not None and all(x is not e[1] for x in ctl.log):
                ctl.log.append(e[1])
        for ch in late:
            ch = ch.copy()
            for i in list(ch.back):
                ch.back[i] = max(ch.back[i], self.slots[i].in_end if self.slots[i].in_end is not None else ch.back[i])
            if ch.s0 is not None:
                ch.s0 = max(ch.s0, c.pos)
            self.changes.append(ch)
            self._keys.add(ch.key)
        ctl.rerun += 1
        return c

    def _run_span(self, D):
        s0 = self.span
        self._span_ops = {id(s.op) for s in self._span_slots()}
        flush = D.flush
        start, end = D.meta.pos, D.meta.pos + len(D)
        base = dict(D.ctx)
        self.buf.append(D)
        ctx = dict(D.ctx)
        outs, cur, first, rewinds = [], start, True, 0
        seg, keep = self._segment()
        while cur < end or first:
            first = False
            cuts = [x.s0 for x in self.changes if x.s0 is not None and cur < x.s0 < end]
            nxt = min([end, (cur//seg + 1)*seg] + [int(round(x)) for x in cuts])
            nxt = max(nxt, cur + 1) if cur < end else end
            self._checkpoint(cur, ctx, len(outs))
            x = self._piece(cur, nxt, ctx, flush and nxt == end, D)
            for i in range(s0, len(self.slots)):
                x = self._run_slot(i, x)
            self._span_ops |= {id(s.op) for s in self._span_slots()}
            outs.append(x)
            ctx = dict(x.ctx)
            late = self._settle()
            if late and rewinds < MAX_REWINDS:
                rewinds += 1
                c = self._rewind(late)
                outs = outs[:c.outs] if c.call == self._call else []
                ctx = {k: copy.deepcopy(v) for k, v in c.ctx.items()}
                cur, first = c.pos, c.pos >= end
                continue
            elif late:
                for ch in late:
                    self._report("late", step=ch.step, fields=dict(ch.fields), at=ch.at, note="too many re-runs in one call")
            cur = nxt
            self._trim(keep)
        out = self._join(outs) if outs else x
        # Upstream state is this call's; the span's own state is what its last piece left
        merged = {k: v for k, v in base.items() if not self._owned(k)}
        merged.update({k: v for k, v in out.ctx.items() if self._owned(k)}) # (a key an op dropped stays dropped)
        merged[FLUSH] = flush
        out.ctx = merged
        if self.emitted is not None and out.meta.pos < self.emitted: # Re-run from an earlier call: what went out already
            cut = int(self.emitted - out.meta.pos)
            meta = out.meta.fork()
            meta.pos = out.meta.pos + min(cut, len(out))
            out = type(out)(np.asarray(out.data)[cut:], out.CAT, out.DT, out.ctx, meta, out.REP)
        self.emitted = out.meta.pos + len(out)
        _ctl.of(self._stream).buffer = sum(len(d) for d in self.buf)
        return out

    def _piece(self, a, b, ctx, flush, D):
        """Items a..b of the span's input, from the re-run buffer"""
        parts = [np.asarray(d.data)[max(a - d.meta.pos, 0):max(b - d.meta.pos, 0)] for d in self.buf
                 if d.meta.pos < b and d.meta.pos + len(d) > a]
        arr = np.concatenate(parts) if parts else np.asarray(D.data)[:0]
        meta = D.meta.fork()
        meta.pos = a
        c = dict(ctx)
        c[FLUSH] = bool(flush)
        return type(D)(arr, D.CAT, D.DT, c, meta, D.REP)

    def _trim(self, keep):
        """Forget checkpoints, and the input before them, further back than a field can reach (the last two stay)"""
        if len(self.ckpts) > 2:
            newest = self.ckpts[-1].pos
            while len(self.ckpts) > 2 and self.ckpts[0].pos < newest - keep:
                self.ckpts.pop(0)
        if self.ckpts:
            oldest = self.ckpts[0].pos
            self.buf = [d for d in self.buf if d.meta.pos + len(d) > oldest] or self.buf[-1:]
