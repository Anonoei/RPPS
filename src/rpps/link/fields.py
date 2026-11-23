"""Fields: what a frame's bits say, as a definition (link headers, frame payloads, a profile step's messages)

A schema is a list of fields, read in order from the bits (most significant bit first unless `order` says otherwise):

    {"name": "length", "bits": 8}                              an unsigned integer
    {"name": "alt", "bits": 12, "signed": true,                two's complement; value = raw*scale + offset
     "scale": 25, "offset": -1000}
    {"name": "type", "bits": 4, "order": "lsb",                bits least significant first; `enum` adds "type.name"
     "enum": {"0": "ADV_IND", "2": "ADV_NONCONN_IND"}}
    {"name": "data", "type": "bytes", "bytes": "length - 1"}   bytes (a count or an expression over the fields before);
                                                               "reverse": true for an address sent low octet first
    {"name": "call", "type": "text", "bytes": 6, "shift": 1}   ASCII, each byte shifted right `shift` bits (AX.25), stripped
    {"name": "pid", "bits": 8, "if": "(control & 1) == 0"}     only if the expression holds over the fields before (per
                                                               frame; a definition's "when" is decided at load, by params)
    {"name": "pcr", "from": 48, "bits": 33}                    at a bit position (from the start of the schema or group)
    {"name": "ads", "fields": [...], "repeat": "*"}             a group: a dict; repeated: a list. repeat: a count, an
    {"name": "addr", "fields": [...], "until": "ext == 1"}      expression, or "*" (until the bits end); until: after each
                                                               element, stop when its fields make the expression true

Expressions are the definitions' (rpps._config.evaluate) over the fields read so far (a group's own first, then its
parents') and `left`, the bits not yet read. A field's `bits` may be an expression too.

parse(bits) gives (values, bits used) and raises Short when the bits end before the schema does (a header waiting for
the rest of its bits); build(values) gives the bits (transmit), fields not given taken as 0 (or their "default").
"""
import numpy as np

from .. import _config
from ..errors import ConfigError

class Short(Exception):
    """The bits ended before the schema did"""

KEYS = {"name", "bits", "from", "order", "signed", "scale", "offset", "enum", "type", "bytes", "reverse", "shift",
        "if", "fields", "repeat", "until", "default", "tx", "comment"}

class Schema:
    def __init__(self, fields, order="msb", name="fields"):
        self.name, self.order = name, order
        if order not in ("msb", "lsb"):
            raise ConfigError(f"{name}: order is msb or lsb, not {order!r}")
        self.fields = [dict(f) for f in fields]
        for f in self._all(self.fields):
            bad = set(f) - KEYS
            if "name" not in f or bad:
                raise ConfigError(f"{name}: a field needs a name and takes {sorted(KEYS)}; got {f}")
            kind = f.get("type", "uint")
            if kind not in ("uint", "bytes", "text"):
                raise ConfigError(f"{name}: field {f['name']}: type is uint, bytes or text, not {kind!r}")
            if "fields" not in f and kind == "uint" and "bits" not in f:
                raise ConfigError(f"{name}: field {f['name']} needs 'bits'")
            if kind != "uint" and "bytes" not in f:
                raise ConfigError(f"{name}: field {f['name']} ({kind}) needs 'bytes'")
            if f.get("order", order) not in ("msb", "lsb"):
                raise ConfigError(f"{name}: field {f['name']}: order is msb or lsb")

    @staticmethod
    def _all(fields):
        for f in fields:
            yield f
            if "fields" in f:
                yield from Schema._all(f["fields"])

    @property
    def top(self):
        """The names of the top-level fields"""
        return [f["name"] for f in self.fields]

    @property
    def fixed_bits(self):
        """Bits of the schema when every field has a fixed size and none is conditional or repeated, else None"""
        n = 0
        for f in self.fields:
            if "fields" in f or "if" in f or "from" in f:
                return None
            size = f.get("bits") if f.get("type", "uint") == "uint" else f.get("bytes")
            if not isinstance(size, int):
                return None
            n += size if f.get("type", "uint") == "uint" else 8*size
        return n

    # ---- receive

    def parse(self, bits, scope=None):
        """(values, bits used) of the schema read from `bits`; raises Short when they end first"""
        b = np.asarray(bits).astype(bool).reshape(-1)
        vals, used = self._group(self.fields, b, 0, dict(scope or {}))
        return vals, used

    def _eval(self, e, scope, at, n):
        if isinstance(e, (int, float)) and not isinstance(e, bool):
            return e
        return _config.evaluate(e, {**scope, "left": n - at}, self.name)

    def _group(self, fields, b, start, outer):
        vals, at = {}, start
        for f in fields:
            scope = {**outer, **_flat(vals)}
            if "if" in f and not self._eval(f["if"], scope, at, len(b)):
                continue
            if "from" in f:
                at = start + int(self._eval(f["from"], scope, at, len(b)))
            name = f["name"]
            if "fields" in f:
                if "repeat" not in f and "until" not in f:
                    vals[name], at = self._group(f["fields"], b, at, scope)
                    continue
                rep = f.get("repeat")
                count = None if rep in (None, "*") else int(self._eval(rep, scope, at, len(b)))
                items = []
                while count is None or len(items) < count:
                    if count is None and "until" not in f and at >= len(b):
                        break # "*": to the end of the bits
                    item, at = self._group(f["fields"], b, at, scope)
                    items.append(item)
                    if "until" in f and self._eval(f["until"], {**scope, **_flat(item)}, at, len(b)):
                        break
                    if len(items) > len(b) + 1:
                        raise ConfigError(f"{self.name}: {name} repeats without reading any bits")
                vals[name] = items
                continue
            kind = f.get("type", "uint")
            if kind == "uint":
                n = int(self._eval(f["bits"], scope, at, len(b)))
            else:
                n = 8*int(self._eval(f["bytes"], scope, at, len(b)))
            if n < 0:
                raise ValueError(f"{self.name}: {name} is {n} bits long")
            if at + n > len(b):
                raise Short(at + n)
            seg = b[at:at + n]
            at += n
            if kind == "uint":
                order = f.get("order", self.order)
                w = 1 << (np.arange(n, dtype=object) if order == "lsb" else np.arange(n - 1, -1, -1, dtype=object))
                raw = int(sum(int(x)*int(y) for x, y in zip(seg, w)))
                if f.get("signed") and n and raw >> (n - 1):
                    raw -= 1 << n
                v = raw*f["scale"] + f.get("offset", 0) if "scale" in f else raw + f.get("offset", 0)
                vals[name] = v
                if "enum" in f:
                    vals[f"{name}.name"] = f["enum"].get(str(raw))
            else:
                data = np.packbits(seg.astype(np.uint8)).tobytes()
                if kind == "bytes":
                    vals[name] = data[::-1] if f.get("reverse") else data
                else:
                    sh = int(f.get("shift", 0))
                    vals[name] = bytes(c >> sh for c in data).decode("ascii", "replace").rstrip(" \0")
        return vals, at

    # ---- transmit

    def build(self, values):
        """The bits of these values (fields not given: their "default", else 0; a conditional field where its "if" holds)"""
        out = []
        self._build(self.fields, dict(values or {}), {}, out)
        return np.concatenate(out).astype(bool) if out else np.zeros(0, bool)

    def _build(self, fields, values, outer, out):
        scope = dict(outer)
        for f in fields:
            name = f["name"]
            if "if" in f and not _config.evaluate(f["if"], {**scope, "left": 0}, self.name):
                continue
            if "from" in f:
                raise ConfigError(f"{self.name}: a schema with 'from' positions is read only")
            if "fields" in f:
                v = values.get(name, [] if ("repeat" in f or "until" in f) else {})
                for item in (v if isinstance(v, list) else [v]):
                    self._build(f["fields"], dict(item), scope, out)
                continue
            kind = f.get("type", "uint")
            v = values.get(name, f.get("default", 0 if kind == "uint" else b""))
            if kind == "uint":
                n = int(_config.evaluate(f["bits"], {**scope, "left": 0}, self.name)) if isinstance(f["bits"], str) else f["bits"]
                raw = round((v - f.get("offset", 0))/f["scale"]) if "scale" in f else int(v) - f.get("offset", 0)
                raw &= (1 << n) - 1
                idx = np.arange(n) if f.get("order", self.order) == "lsb" else np.arange(n - 1, -1, -1)
                out.append(np.array([(raw >> int(i)) & 1 for i in idx], dtype=bool))
                scope[name] = v
            else:
                nb = f["bytes"] if isinstance(f["bytes"], int) else int(_config.evaluate(f["bytes"], {**scope, "left": 0}, self.name))
                if kind == "text":
                    sh = int(f.get("shift", 0))
                    data = bytes((c << sh) & 0xFF for c in str(v).ljust(nb).encode("ascii")[:nb])
                else:
                    data = bytes.fromhex(v) if isinstance(v, str) else bytes(v)
                    data = data[::-1] if f.get("reverse") else data
                    data = data[:nb].ljust(nb, b"\0")
                out.append(np.unpackbits(np.frombuffer(data, np.uint8)).astype(bool))
                scope[name] = v

def _flat(vals):
    """The values a later expression can name: a group's scalars (lists and dicts are not numbers)"""
    return {k: v for k, v in vals.items() if not isinstance(v, (list, dict))}
