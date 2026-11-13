"""Header-driven control: the control path that runs back up a chain (README.dev.md, Header-driven control)

A header step (a physical header, a link header) and a frame sync publish what they decoded, with the absolute stream
position it applies from. A profile's `controls` maps those fields, through an optional table, to parameters of other
steps; a runner changes each parameter at the matching position of that step's input.

What operators call (both directions):

    publish(meta, op, {"modcod": 12, "pilots": 1}, at=<input position>, out=<index in this call's output>, span=None,
            group=None)
        at: absolute position in the op's input from which the fields apply. out: where that is in what the op gives in
        this call (None: nothing it gives is affected). span: input items the fields hold for, after which they go back
        to the controls' defaults (a PLCP rate holds for its payload only). group: publications made from one header
        share it (default: `at`), so a runner that re-runs keeps them together
    anchor(meta, op, at, out, rate=None)
        a step that drops or inserts items says which output item an input position became, so positions translate
        exactly through it

The profile side (`"controls"` in a profile, keyed by the header step's id):

    "controls": {
      "pls": {"table": "dvbs2/modcod", "key": "modcod",           a table resource: one row per key value
              "default": {"modcod": 4, "pilots": 1},              field values before any header, and after a span
              "schedule": [{"modcod": 4}, {"modcod": 12}],        transmit: the fields of each frame in turn
              "frame": {"calc": "kbch - 80"},                     transmit: items at the first step per frame
              "reach": 64,                                        how far back (header input items) a field can apply
              "drive": {"map": {"name": "constellation"},         step id -> {parameter: source}: a field, a column of
                        "ldpc": {"name": "ldpc"},                   the key's row, or {"calc": ...} over both
                        "plscram": {"length": {"calc": "slots*90 + pilots*((slots - 1)//16)*36"}}}}}

A "calc" is an expression (as `{"expr": ...}` takes) worked out when the fields arrive; an "expr" or a "param" in a profile
is worked out when it is loaded, from the profile's parameters (a schedule of `{"modcod": {"param": "modcod"}}`).

A step's parameters are fixed unless controlled. Each step declares how it takes a change, per parameter (`Step.controls`):
LIVE (in place, keeping its state) or RESTART (rebuilt from the position: what was in flight is discarded and reported).
A parameter it does not declare cannot be controlled, and the loader refuses a profile that tries. A change applied to a
step marks a frame start for it, so frame-scoped steps (a header remover, a pilot remover) restart their frame there.
"""
import bisect
import functools

from .. import _config
from ..errors import ConfigError

LIVE, RESTART = "live", "restart"

# ---- What operators call

class Control:
    """The control path of one stream (`stream.control`)

    published, anchors: what operators reported since the runner last collected them (the runner empties these)
    log: every change the runner accepted, in order: {"step", "fields", "at", "targets"}
    reports: what did not go as announced: late changes re-run or dropped, restarts, fields with no table row
    rerun: re-runs done; buffer: the re-run buffer's current size, in items at its first step
    """
    __slots__ = ("published", "anchors", "log", "reports", "rerun", "buffer")
    def __init__(self):
        self.published, self.anchors, self.log, self.reports = [], [], [], []
        self.rerun, self.buffer = 0, 0

    def fields(self, step=None):
        """The field sets published so far, oldest first: [(step id, fields, position)]"""
        return [(e["step"], e["fields"], e["at"]) for e in self.log if step is None or e["step"] == step]

def of(stream):
    """The stream's Control, made on first use"""
    if stream.control is None:
        stream.control = Control()
    return stream.control

def publish(meta, op, fields, at, out=None, span=None, group=None):
    """Fields an op decoded (or was given, transmitting), applying from absolute input position `at` onward (see module)"""
    of(meta.stream).published.append((op, dict(fields), int(at), None if out is None else int(out),
                                      None if span is None else int(span), int(at) if group is None else group))

def anchor(meta, op, at, out, rate=None):
    """Absolute input position `at` became item `out` of this call's output (a step that drops or inserts items)"""
    of(meta.stream).anchors.append((op, int(at), int(out), rate))

# ---- Tables

class Table:
    """A table resource: rows of named columns, looked up by the key fields

        {"base": "table", "comment": "...", "key": "modcod",                 or a list of key columns
         "columns": ["modcod", "constellation", "ldpc"],
         "rows": [[4, "dvbs2_qpsk", "ldpc/dvbs2_64800_1_2"], ...]}
    """
    def __init__(self, name, key, columns, rows):
        self.name = name
        self.key = [key] if isinstance(key, str) else list(key)
        self.columns = list(columns)
        missing = [k for k in self.key if k not in self.columns]
        if missing:
            raise ConfigError(f"table {name}: key {missing} is not a column of {self.columns}")
        self.rows = {}
        for r in rows:
            if len(r) != len(self.columns):
                raise ConfigError(f"table {name}: row {r} has {len(r)} values for {len(self.columns)} columns")
            d = dict(zip(self.columns, r))
            self.rows[tuple(_norm(d[k]) for k in self.key)] = d

    def row(self, fields):
        """The row the fields' key values pick, or None (a key field missing, or no such row)"""
        if any(k not in fields or fields[k] is None for k in self.key):
            return None
        return self.rows.get(tuple(_norm(fields[k]) for k in self.key))

    def __repr__(self):
        return f"<Table {self.name}: {len(self.rows)} rows by {self.key}>"

    @staticmethod
    def load(name, obj):
        for k in ("key", "columns", "rows"):
            if k not in obj:
                raise ConfigError(f"table {name}: needs '{k}'")
        return Table(name, obj["key"], obj["columns"], obj["rows"])

def _norm(v):
    return int(v) if isinstance(v, bool) else v

DOMAIN = _config.register_domain("table", {"table": Table.load})

@functools.lru_cache(maxsize=None)
def table(name):
    """Load a table resource by name: `rp.profile.table("dvbs2/modcod")`"""
    return _config.load("table", name)

# ---- Controls entries

_NA = object()

class Entry:
    """One entry of a profile's `controls`: the fields of one header step and the parameters they drive (see module)"""
    def __init__(self, header, obj, where):
        self.header = header
        if not isinstance(obj, dict) or "drive" not in obj:
            raise ConfigError(f"{where}: controls[{header!r}] needs 'drive': {{step id: {{parameter: source}}}}")
        self.table = table(obj["table"]) if obj.get("table") else None
        self.defaults = dict(obj.get("default", {}))
        self.schedule = [dict(f) for f in obj.get("schedule", [])]
        self.frame = obj.get("frame")
        self.reach = obj.get("reach")
        self.drive = {sid: dict(ps) for sid, ps in obj["drive"].items()}

    def __repr__(self):
        return f"<Entry {self.header} -> {sorted(self.drive)}>"

    def names(self, fields):
        """Fields and the columns of their row, fields first; and the row (None without a table or a key)"""
        row = self.table.row(fields) if self.table is not None else None
        return {**(row or {}), **fields}, row

    def values(self, fields):
        """{step id: {parameter: value}} these fields set, and the row they picked (None: none). A parameter whose source
        the fields don't give is left alone, so a change that carries only some fields changes only what they drive"""
        names, row = self.names(fields)
        out = {}
        for sid, ps in self.drive.items():
            for p, src in ps.items():
                v = self._value(src, names, fields)
                if v is not _NA:
                    out.setdefault(sid, {})[p] = v
        return out, row

    @staticmethod
    def _value(src, names, fields):
        if isinstance(src, str):
            return names[src] if src in names else _NA
        if isinstance(src, dict) and "calc" in src:
            try:
                return _config.evaluate(src["calc"], names, "controls")
            except (ConfigError, TypeError):
                return _NA # A name it needs is not among these fields
        return src # A constant

    def frame_items(self, fields):
        """Transmit: items of the first step one frame takes, from `frame` (None: one frame per call)"""
        if self.frame is None:
            return None
        v = self._value(self.frame, self.names(fields)[0], fields)
        if v is _NA or v is None:
            raise ConfigError(f"controls[{self.header!r}]: frame {self.frame} not given by fields {fields}")
        return int(v)

    def revert(self, fields):
        """The default values of these fields: what holds after a span"""
        return {k: self.defaults.get(k) for k in fields}

def parse(obj, steps, where):
    """The controls section of a profile, checked against its steps: {header step id: Entry}

    Every step it names must exist once, and every parameter it drives must be one the step can take, live or restart"""
    if not obj:
        return {}
    by_id = {}
    for st in steps:
        by_id.setdefault(st.id, []).append(st)
    def one(sid, what):
        hits = by_id.get(sid, [])
        if len(hits) != 1:
            raise ConfigError(f"{where}: controls {what} {sid!r}: {'no such step' if not hits else 'several steps have that id'}"
                              f" (step ids: {sorted(by_id)})")
        return hits[0]
    out = {}
    for header, e in obj.items():
        one(header, "header")
        entry = Entry(header, e, where)
        for sid, ps in entry.drive.items():
            st = one(sid, "drives")
            for p in ps:
                how = st.controls.get(p)
                if how not in (LIVE, RESTART):
                    raise ConfigError(f"{where}: controls[{header!r}] drives {sid}.{p}, which {st.category}:{st.kind} cannot "
                                      f"take (it takes {sorted(st.controls) or 'no controlled parameters'})")
        out[header] = entry
    return out

# ---- Positions through a step

class PosMap:
    """How a step's input positions become its output positions: anchors (input, output, rate), from which the output
    advances `rate` items per input item. A step gets one when it first runs, one at every change applied to it (its rate
    may change), and one wherever it reports that it dropped or inserted items"""
    __slots__ = ("a",)
    def __init__(self, anchors=None):
        self.a = list(anchors or [])

    def copy(self):
        return PosMap([list(x) for x in self.a])

    def __bool__(self):
        return bool(self.a)

    def cut(self, i):
        """Forget the anchors after input position i (they came from a pass that is being re-run)"""
        self.a = [x for x in self.a if x[0] <= i]

    def add(self, i, o, rate):
        """An anchor; one at the same input position is replaced (what a step reports wins over what was worked out)"""
        k = bisect.bisect_left([x[0] for x in self.a], i)
        if k < len(self.a) and self.a[k][0] == i:
            self.a[k] = [i, o, rate]
        else:
            self.a.insert(k, [i, o, rate])

    def rate_at(self, i):
        k = bisect.bisect_right([x[0] for x in self.a], i) - 1
        return self.a[max(k, 0)][2] if self.a else 1.0

    def fwd(self, i):
        """Output position of input position i"""
        if not self.a:
            return i
        k = max(bisect.bisect_right([x[0] for x in self.a], i) - 1, 0)
        a = self.a[k]
        return a[1] + (i - a[0])*a[2]

    def back(self, o):
        """Input position of output position o"""
        if not self.a:
            return o
        k = max(bisect.bisect_right([x[1] for x in self.a], o) - 1, 0)
        a = self.a[k]
        return a[0] + (o - a[1])/a[2] if a[2] else a[0]

def rate_of(op, how):
    """Output items per input item of an operator, in the direction it runs (how: "+" stage, "/" decode, "*" encode).
    From the operator's `pos_rate(decode)` when it has one; None when it can't say (the runner measures it)"""
    f = getattr(op, "pos_rate", None)
    if f is None:
        return None
    r = f(how != "*")
    return None if r is None else float(r)
