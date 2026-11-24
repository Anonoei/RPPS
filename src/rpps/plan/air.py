"""Plans from what a forward link says: a definition (`"base": "builder"`, config/plan/<family>/<name>.json) maps the
tables a forward link carries onto the plan model, and its clock reference packets onto the capture

    b = rp.plan.builder("dvbrcs2/forward")
    for m in forward.take_messages():   # the forward profile's messages: tables, clock reference packets, frame starts
        b.feed(m)
    b.plan                              # rp.plan.Plan, its assignments those of the tables seen
    b.clock(t)                          # the network clock at sample t of the forward capture (the packets, fitted)
    b.tables(plan, superframes=[...])   # transmit: [(protocol, PDU)] of a plan's tables
    b.reference(ncr)                    # transmit: (protocol, PDU) of one clock reference

The definition:

    "clock": 27e6                       the plan's ticks a second
    "tables": {"header": "rcs2_table",  the header (config/header/link) that reads each table PDU; its messages' `kind`,
               "kind": "table",         the field naming the table and each table's value of it, the protocol the PDUs
               "id": "table_id",        go in on transmit, and the head fields on transmit (expressions over
               "ids": {"sct": 160, ...},network_id and version)
               "protocol": 130,
               "tx": {"interactive_network_id": "network_id", ...}}
    "sections": {"tx_types": {...}, "frame_types": {...}, "sequences": {...}, "assignments": {...}}
    "reference": {...}                  the clock (below)

A section maps a table onto the plan definition's key of the same name (rpps.plan's model, as `rp.plan.load` reads it):

    {"table": "bct", "each": "tx_types",    the table, and the list in it whose items are the section's entries (none:
     "if": "tx_format_class == 1",          the whole table is one); only the items the expression holds for
     "rx": {"id": "tx_type", "btus": "tx_block_size", "fields": {"bits": "payload_size*8"}},
     "tx": {"tx_type": "id", "tx_block_size": "btus", "payload_size": "bits//8"},
     "defaults": {"bits": 0}}

`rx` is the entry as a template: a string is an expression over the item's fields (its groups' fields too, by name: a
sized group's, a nested group's; the enclosing items' as well), a number itself, a list each of its own, and
{"each": "<a list of the item>", "rx": template} a list of that list's items. tx_types, frame_types and sequences are
keyed by the entry's "id"; an assignments entry is {sequence, superframe, frame, offset, slots: [[id, tx type]]}, and
"count_bits" says how many of the superframe count's bits the table carries (counted on from the sequence's count).
`tx` is the reverse: table fields from expressions over the entry (nested keys dotted: `btu.duration`; a list's
element `it` when not an object), with {"each": "<a list of the entry>", "tx": {...}} making a list. An enum field is
set by name (`"modulation_scheme.name": "constellation"`) and a field the header's `tx` computes (a loop count) is left
out. `defaults` fill names an entry or item lacks, both ways. On transmit a tx type's `word` (a sequence name) is there
as its bits (`[]` when it has none).

The clock reference: packets of the network clock, each the clock at a frame start (an event) `lag` events before the
one whose frame carries it:

    "reference": {"kind": "gse", "if": "protocol == 129", "protocol": 129, "header": "rcs2_ncr",
                  "rx": "base*300 + ext", "tx": {"base": "(value//300) % (1 << 33)", "ext": "value % 300"},
                  "wrap": "(1 << 33)*300", "events": "pls", "lag": 2}

The builder pairs each packet (a message of `kind` the `if` holds for; its payload read by `header`, `rx` its value)
with the event (messages of kind `events`) `lag` before the last one at or before it, and fits clock = a + b * sample by
least squares, the `wrap` undone.
"""
import numpy as np

from .. import _config
from ..errors import ConfigError
from . import Plan, COUNT_MOD

SECTIONS = ("tx_types", "frame_types", "sequences", "assignments")


def _bytes(bits):
    bits = np.asarray(bits, bool)
    return np.packbits(np.concatenate((bits, np.zeros(-len(bits) % 8, bool)))).tobytes()

def _scope(item, outer=None):
    """The names an expression over a table item sees: its scalars, its groups' (nested, by name), the enclosing items'"""
    out = dict(outer or {})
    def walk(d):
        groups = []
        for k, v in d.items():
            if isinstance(v, dict):
                groups.append(v)
            elif not isinstance(v, list):
                out[k] = v
        for g in groups: # (the item's own fields first, then its groups')
            for k, v in g.items():
                if not isinstance(v, (dict, list)):
                    out.setdefault(k, v)
            walk({k: v for k, v in g.items() if isinstance(v, dict)})
    walk(item)
    return out

def _lists(item):
    """The lists of an item and of its groups, by name"""
    out = {}
    for k, v in item.items():
        if isinstance(v, list):
            out[k] = v
        elif isinstance(v, dict):
            for kk, vv in _lists(v).items():
                out.setdefault(kk, vv)
    return out

def _dotted(entry, prefix=""):
    """An entry's values by dotted name (`btu.duration`), its lists whole"""
    out = {}
    for k, v in entry.items():
        if isinstance(v, dict) and k != "fields":
            out.update(_dotted(v, f"{prefix}{k}."))
        elif isinstance(v, dict): # (a tx type's fields: by their own names)
            out.update(v)
        else:
            out[f"{prefix}{k}"] = v
    return out


class Builder:
    """A plan and a clock from a forward link's messages (see the module). `plan` is None until every keyed section's
    table is in"""
    def __init__(self, name, obj):
        self.name = name
        self.ticks = float(obj.get("clock", 1.0))
        t = obj.get("tables") or {}
        for need in ("header", "kind", "id", "ids"):
            if need not in t:
                raise ConfigError(f"plan builder {name}: tables needs {need!r}")
        self.table = t
        self.sections = dict(obj.get("sections") or {})
        bad = set(self.sections) - set(SECTIONS)
        if bad:
            raise ConfigError(f"plan builder {name}: sections are {SECTIONS}, not {sorted(bad)}")
        for k, s in self.sections.items():
            if s.get("table") not in t["ids"]:
                raise ConfigError(f"plan builder {name}: section {k} reads table {s.get('table')!r}, not one of {sorted(t['ids'])}")
            if k != "assignments" and "each" not in s:
                raise ConfigError(f"plan builder {name}: section {k} needs 'each' (the table's list of its entries)")
        self.ref = obj.get("reference")
        self._schema = self._ref_schema = None
        self.tables_in = {}                     # {table: content}, the last of each
        self.assigned = []                      # assignment tables' contents, in order
        self.events, self.packets = [], []      # sample of each event; (sample, clock value) of each packet
        self._plan, self._fit = None, None

    @staticmethod
    def load(name, obj):
        return Builder(name, obj)

    def __repr__(self):
        return f"<plan Builder {self.name}: tables {sorted(self.tables_in)}, {len(self.packets)} clock reference(s)>"

    def _eval(self, e, scope, what):
        return _config.evaluate(e, scope, f"plan builder {self.name}: {what}")

    @property
    def schema(self):
        if self._schema is None:
            self._schema = _header_schema(self.table["header"])
        return self._schema

    # ---- receive

    def feed(self, m):
        """Take one message of the forward link's: a table, a clock reference packet or an event"""
        f = m.fields
        r = self.ref
        if m.kind == self.table["kind"] and self.table["id"] in f:
            if m.crc_ok is False:
                return
            names = {v: k for k, v in self.table["ids"].items()}
            tab = names.get(f[self.table["id"]])
            content = (f.get("payload") or {}).get(tab)
            if tab is None or content is None:
                return
            if any(s["table"] == tab for k, s in self.sections.items() if k == "assignments"):
                self.assigned.append(content)
                if self._plan is not None:
                    self._assign(self._plan, content)
            else:
                self.tables_in[tab], self._plan = content, None
        elif r is not None and m.kind == r.get("events"):
            self.events.append(m.sample)
        elif r is not None and m.kind == r["kind"] and m.payload and m.crc_ok is not False and \
                ("if" not in r or self._eval(r["if"], {k: v for k, v in f.items() if not isinstance(v, (dict, list))}, "reference if")):
            self.packets.append((m.sample, self.value(m.payload)))
            self._fit = None

    # ---- the clock

    @property
    def ref_schema(self):
        if self._ref_schema is None:
            if self.ref is None:
                raise ConfigError(f"plan builder {self.name}: no clock reference")
            self._ref_schema = _header_schema(self.ref["header"])
        return self._ref_schema

    def value(self, payload):
        """The clock value of one reference packet's payload"""
        b = np.unpackbits(np.frombuffer(bytes(payload), np.uint8)).astype(bool)
        v, _ = self.ref_schema.parse(b)
        return self._eval(self.ref["rx"], _scope(v), "reference rx")

    def pairs(self):
        """(sample of the event, clock value) of each reference packet: the event `lag` before the one carrying it"""
        ev = np.sort(np.asarray(self.events, dtype=np.float64))
        lag, out = int(self.ref.get("lag", 0)), []
        for s, v in self.packets:
            k = int(np.searchsorted(ev, s, side="right")) - 1 # (the event whose frame carries the packet)
            if k - lag >= 0:
                out.append((ev[k - lag], v))
        return out

    def fit(self):
        """(a, b): clock = a + b * forward sample, least squares over the pairs (the clock's wrap undone); None with
        fewer than two"""
        if self._fit is None:
            pr = self.pairs()
            if len(pr) < 2:
                return None
            x = np.array([p[0] for p in pr])
            y = np.array([p[1] for p in pr], dtype=np.float64)
            if "wrap" in self.ref:
                y = np.unwrap(y, period=float(self._eval(self.ref["wrap"], {}, "reference wrap")))
            b, a = np.polyfit(x, y, 1)
            self._fit = (a, b)
        return self._fit

    def clock(self, sample):
        """The clock at a forward capture sample (a time the return capture shares: its first sample's clock is the plan's
        origin there)"""
        fit = self.fit()
        return None if fit is None else fit[0] + fit[1]*sample

    # ---- the plan

    @property
    def plan(self):
        need = {s["table"] for k, s in self.sections.items() if k != "assignments"}
        if self._plan is None and need <= set(self.tables_in):
            self._plan = Plan.load(self.name, self.definition())
            for t in self.assigned:
                self._assign(self._plan, t)
        return self._plan

    def definition(self):
        """The plan definition (rpps.plan's model) of the tables in, without assignments"""
        out = {"base": "plan", "clock": self.ticks}
        for key, s in self.sections.items():
            if key == "assignments" or s["table"] not in self.tables_in:
                continue
            out[key] = {}
            for e in self._rx(s, self.tables_in[s["table"]]):
                if "id" not in e:
                    raise ConfigError(f"plan builder {self.name}: section {key}'s rx gives no 'id'")
                out[key][str(int(e.pop("id")))] = e
        return out

    def _rx(self, s, content):
        """A section's entries of one table's content"""
        top = {**s.get("defaults", {}), **_scope(content)}
        items = [(content, {})] if "each" not in s else [(it, top) for it in content.get(s["each"], [])]
        out = []
        for it, outer in items:
            sc = _scope(it, outer) if outer else top
            sc = {**s.get("defaults", {}), **sc}
            if "if" in s and not self._eval(s["if"], sc, f"{s['table']} if"):
                continue
            e = self._template(s["rx"], it, sc, s)
            out.extend(e if isinstance(e, list) else [e])
        return out

    def _template(self, t, item, sc, s):
        if isinstance(t, dict) and "each" in t:
            lst = _lists(item).get(t["each"])
            if lst is None:
                raise ConfigError(f"plan builder {self.name}: {s['table']} has no list {t['each']!r}")
            return [self._template(t["rx"], x, {**s.get("defaults", {}), **_scope(x, sc)}, s) for x in lst]
        if isinstance(t, dict):
            return {k: self._template(v, item, sc, s) for k, v in t.items()}
        if isinstance(t, list):
            return [self._template(v, item, sc, s) for v in t]
        if isinstance(t, str):
            return self._eval(t, sc, f"{s['table']} rx")
        return t

    def _assign(self, plan, content):
        s = self.sections["assignments"]
        bits = s.get("count_bits")
        for e in self._rx(s, content):
            seq = plan.sequences.get(int(e["sequence"]))
            if seq is None:
                continue
            count = int(e["superframe"])
            if bits: # (the table's low bits: the sequence's count, onwards)
                count = seq.count + (count - seq.count) % (1 << int(bits))
            plan.assign(seq.id, count, e["frame"], [tuple(x) for x in e["slots"]], e.get("offset", 0))

    # ---- transmit

    def tables(self, plan, network_id=1, version=0, superframes=()):
        """[(protocol, PDU bytes)] of a plan's tables: each keyed section's, then the assignments' of each (sequence,
        superframe count) in `superframes` that the plan has assignments for"""
        d = plan.definition()
        head = {k: self._eval(v, {"network_id": network_id, "version": version}, "tables tx") if isinstance(v, str) else v
                for k, v in (self.table.get("tx") or {}).items()}
        proto = int(self.table.get("protocol", 0))
        contents = {}
        for key, s in self.sections.items():
            if key == "assignments":
                continue
            entries = [{"id": int(k), **v} for k, v in d.get(key, {}).items()]
            if key == "tx_types":
                for e in entries:
                    e["word"] = _word(e.get("fields", {}).get("word"))
            contents.setdefault(s["table"], {})[s["each"]] = [self._tx(s["tx"], e, s) for e in entries]
        out = [self._pdu(t, contents[t], head) for t in self.table["ids"] if t in contents]
        s = self.sections.get("assignments")
        for seq, count in superframes if s else ():
            given = plan.assignments.get((int(seq), int(count) % COUNT_MOD))
            if not given:
                continue
            e = {"sequence": int(seq), "superframe": int(count) % COUNT_MOD,
                 "frames": [{"frame": f, "offset": off, "slots": [list(x) for x in items]} for f, (off, items) in sorted(given.items())]}
            out.append(self._pdu(s["table"], self._tx(s["tx"], e, s), head))
        return [(proto, pdu) for pdu in out]

    def _pdu(self, table, content, head):
        return _bytes(self.schema.build({**head, self.table["id"]: self.table["ids"][table], table: content}))

    def _tx(self, t, entry, s):
        """Table fields of one entry (a list's element: `it`, when not an object)"""
        sc = {**s.get("defaults", {}), **(_dotted(entry) if isinstance(entry, dict) else {"it": entry})}
        out = {}
        for k, v in t.items():
            if isinstance(v, dict) and "each" in v:
                lst = entry.get(v["each"], []) if isinstance(entry, dict) else []
                out[k] = [self._tx(v["tx"], x, s) for x in lst]
            elif isinstance(v, dict):
                out[k] = self._tx(v, entry, s)
            elif isinstance(v, str):
                out[k] = self._eval(v, sc, f"{s['table']} tx")
            else:
                out[k] = v
        return _nested(out)

    def reference(self, value):
        """(protocol, PDU bytes) of the clock reference packet of a clock value"""
        r = self.ref
        if r is None:
            raise ConfigError(f"plan builder {self.name}: no clock reference")
        v = {k: self._eval(e, {"value": int(round(value))}, "reference tx") if isinstance(e, str) else e
             for k, e in r["tx"].items()}
        return int(r.get("protocol", 0)), _bytes(self.ref_schema.build(v))


def _nested(fields):
    """Table fields with dotted group names (`tx_format.lm.payload_size`) as nested groups"""
    out = {}
    for k, v in fields.items():
        parts = k.split(".")
        if len(parts) > 1 and parts[-1] == "name": # (an enum field by name: `modulation_scheme.name`)
            parts = parts[:-2] + [f"{parts[-2]}.name"]
        d = out
        for p in parts[:-1]:
            d = d.setdefault(p, {})
        d[parts[-1]] = v
    return out

def _word(name):
    """A sequence's bits by name ([] for none)"""
    if not name:
        return []
    from ..sequence import load as seq
    return [int(b) for b in np.asarray(seq(name).bits, bool)]

def _header_schema(name):
    from ..link.fields import Schema
    obj = _config.read(_config.find("header", name, "link"))
    return Schema(obj["fields"] + (obj.get("payload_fields") or []), obj.get("order", "msb"), f"header {name}")
