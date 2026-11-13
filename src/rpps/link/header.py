"""Link headers: fields, a CRC, and the payload they announce (ARCHITECTURE.md, link and Header-driven control)

A definition, config/header/link/<name>.json, `rp.link.load("header", "ieee80211b_plcp")`:

    {"base": "header", "type": "link", "comment": "...", "ref": "...",
     "order": "lsb",                                      field bits sent least ("lsb") or most ("msb") significant first
     "fields": [{"name": "signal", "bits": 8},
                {"name": "length", "bits": 16,
                 "tx": "(bits*10 + signal - 1)//signal"},   transmit: its value from the payload's bits and the other
                ...],                                         fields (an expression, as "expr" takes, worked out per frame);
                                                              else the controlled value, its "default", or 0
     "crc": "crc16_genibus",                              a crc definition over the field bits, sent after them
     "payload": "length*signal//10",                      receive: payload bits after the header, from the fields
     "preamble": {"ones": 128, "word": "0xF3A0", "bits": 16},   transmit: sent before the header (in `order`)
     "framed": true}                                      frames of `length` bits (a controlled parameter: DVB-S2 Kbch),
                                                          the payload padded to fill them

Receiving, each frame starts where a profile applies a change (a delimiter before it publishes frame starts), or, framed,
`length` bits after the last. The header's fields, with "bits" (the payload's length), are published from the payload's
first bit for the payload's length, and {"frame_bits": header + payload} from the frame's first bit (rp.profile.publish),
so a profile can drive the rate of the payload and the length of the frame. A header whose CRC fails is counted, and its
frame dropped. Out come the payloads.
Transmitting, a frame starts at each change a profile applies: its fields are the controlled values; framed, its payload
is `length` minus the header, else everything until the next frame start (one frame per call).
"""
from types import SimpleNamespace

import numpy as np

from .. import _config
from ..base import Digital, dtype
from ..base.rpps import Pipe, FRAME
from ..base.types import Category, Rep
from ..base.data import expect
from ..errors import ConfigError

class Header(Pipe):
    CATEGORY = Category.LINK
    SCOPE, SCOPE_NEEDS = FRAME, ("length",) # A frame's own header and payload, when a control gives its length

    def __init__(self, name, fields, order="msb", crc=None, payload=None, preamble=None, framed=False):
        self.name = name
        self.fields = [dict(f) for f in fields]
        for f in self.fields:
            if "name" not in f or "bits" not in f:
                raise ConfigError(f"header {name}: a field needs 'name' and 'bits', got {f}")
        self.order = order
        self.crc = None if crc is None else _crc(crc)
        self.payload = payload
        self.framed = bool(framed)
        self.field_bits = sum(int(f["bits"]) for f in self.fields)
        self.header_bits = self.field_bits + (self.crc.width if self.crc is not None else 0)
        self.preamble = np.zeros(0, dtype=bool) if not preamble else np.concatenate((
            np.ones(int(preamble.get("ones", 0)), dtype=bool),
            self._value_bits(preamble["word"], int(preamble["bits"])) if "word" in preamble else np.zeros(0, bool)))
        self.values = {f["name"]: f.get("default", 0) for f in self.fields}
        self.length = None
        self._start = True
        self.CONTROLS = {"frame": "live", "length": "live", **{f["name"]: "live" for f in self.fields}}

    def __str__(self):
        return f"Header:{self.name}"

    def takes(self, encode):
        return {Rep.DATA, Rep.PACKETS} if encode else {Rep.DATA}

    def pos_rate(self, decode):
        if decode or not self.framed or not self.length:
            return 1.0
        return self.length/(self.length - self.header_bits)

    # ---- control

    def control(self, frame=None, length=None, **fields):
        """A frame starts here (receive and transmit); length: its bits, framed; fields: transmit values"""
        unknown = set(fields) - set(self.values)
        if unknown:
            raise ConfigError(f"header {self.name}: no field {sorted(unknown)}")
        if length is not None:
            self.length = int(length)
        self.values = {**self.values, **{k: v for k, v in fields.items() if v is not None}}
        self._start = True

    # ---- bits

    def _value_bits(self, v, n):
        v = int(v, 0) if isinstance(v, str) else int(v)
        idx = np.arange(n) if self.order == "lsb" else np.arange(n - 1, -1, -1)
        return ((v >> idx) & 1).astype(bool)

    def _value(self, bits):
        b = np.asarray(bits, dtype=np.int64)
        w = 1 << (np.arange(len(b)) if self.order == "lsb" else np.arange(len(b) - 1, -1, -1))
        return int(b @ w)

    def parse(self, bits):
        """(fields, CRC ok) of a header's bits"""
        bits = np.asarray(bits).astype(bool)
        out, i = {}, 0
        for f in self.fields:
            n = int(f["bits"])
            out[f["name"]] = self._value(bits[i:i + n])
            i += n
        ok = True
        if self.crc is not None:
            ok = self.crc.from_bits(bits[i:i + self.crc.width]) == self.crc.compute_bits(bits[:i])
        return out, ok

    def build(self, values):
        """A header's bits for these field values"""
        bits = np.concatenate([self._value_bits(int(values.get(f["name"], 0)), int(f["bits"])) for f in self.fields])
        if self.crc is not None:
            bits = np.concatenate((bits, self.crc.to_bits(self.crc.compute_bits(bits))))
        return bits

    def _tx_values(self, nbits):
        v = dict(self.values)
        for f in self.fields:
            if "tx" in f:
                v[f["name"]] = int(_config.evaluate(f["tx"], {**v, "bits": nbits}, f"header {self.name}"))
        return v

    def payload_bits(self, fields):
        if self.payload is None:
            return None
        return int(_config.evaluate(self.payload, fields, f"header {self.name}"))

    # ---- receive

    def _state(self, ctx, enc):
        key = (self, "tx" if enc else "rx")
        st = ctx.get(key)
        st = SimpleNamespace(**vars(st)) if st is not None else SimpleNamespace(
            mode="header", held=np.zeros(0, bool), start=None, left=0, frame_left=None, pend=np.zeros(0, bool), pend_pos=None)
        ctx[key] = st
        return st

    def decode(self, data):
        from ..profile.control import publish, anchor
        ctx = dict(data.ctx)
        st = self._state(ctx, False)
        x = data.bits() if hasattr(data, "bits") else np.asarray(data.data)
        x = np.asarray(x).astype(bool)
        pos0 = data.meta.pos
        est = data.meta.stream.est.setdefault(self.name, {"frames": 0, "crc_failed": 0, "last": None})
        out, n_out, i = [], 0, 0
        if self._start:
            self._start = False
            self._restart(st, pos0)
        while i < len(x):
            if self.framed and self.length and st.frame_left is not None and st.frame_left <= 0:
                self._restart(st, pos0 + i) # Framed: the next frame follows this one
            if st.mode == "header":
                take = min(self.header_bits - len(st.held), len(x) - i)
                st.held = np.concatenate((st.held, x[i:i + take]))
                i += take
                self._used(st, take)
                if len(st.held) < self.header_bits:
                    break
                fields, ok = self.parse(st.held)
                if not ok:
                    est["crc_failed"] += 1
                    st.mode = "skip"
                    continue
                n = self.payload_bits(fields)
                if self.framed and self.length:
                    n = min(n if n is not None else self.length, self.length - self.header_bits)
                est["frames"] += 1
                est["last"] = fields
                at = st.start + self.header_bits
                publish(data.meta, self, {**fields, "bits": n}, at=at, out=n_out, span=n, group=st.start)
                publish(data.meta, self, {"frame_bits": self.header_bits + n}, at=st.start, out=None,
                        span=self.header_bits + n, group=st.start)
                anchor(data.meta, self, at, n_out)
                st.mode, st.left = "payload", n
            elif st.mode == "payload":
                take = min(st.left, len(x) - i)
                out.append(x[i:i + take])
                n_out += take
                i += take
                st.left -= take
                self._used(st, take)
                if st.left <= 0:
                    st.mode = "skip"
            else: # Skip to the next frame
                take = len(x) - i if st.frame_left is None else min(len(x) - i, max(st.frame_left, 0))
                i += take
                self._used(st, take)
                if st.frame_left is None:
                    break
        y = np.concatenate(out) if out else np.zeros(0, bool)
        return data.derive(y, self.CATEGORY, dtype.BITS, Digital, ctx, rep=Rep.PACKETS)

    def _restart(self, st, pos):
        st.mode, st.held, st.start, st.left = "header", np.zeros(0, bool), pos, 0
        st.frame_left = self.length if self.framed and self.length else None

    @staticmethod
    def _used(st, n):
        if st.frame_left is not None:
            st.frame_left -= n

    # ---- transmit

    def encode(self, data):
        from ..profile.control import publish, anchor
        ctx = dict(data.ctx)
        st = self._state(ctx, True)
        x = data.bits() if hasattr(data, "bits") else np.asarray(data.data)
        x = np.asarray(x).astype(bool).reshape(-1)
        pos = data.meta.pos
        out, n_out = [], 0
        def frame(payload, at):
            nonlocal n_out
            v = self._tx_values(len(payload))
            pad = np.zeros(0, bool)
            if self.framed and self.length:
                pad = np.zeros(self.length - self.header_bits - len(payload), bool)
            head = np.concatenate((self.preamble, self.build(v)))
            anchor(data.meta, self, at, n_out)
            publish(data.meta, self, {**v, "bits": len(payload)}, at=at, out=n_out + len(head), span=len(payload))
            out.append(np.concatenate((head, payload, pad)))
            n_out += len(head) + len(payload) + len(pad)
        if not self.framed or not self.length: # One frame per call, from a frame start
            if self._start or len(x):
                self._start = False
                frame(x, pos)
        else:
            if self._start:
                self._start = False
                if len(st.pend): # A frame cut short by the next one: out padded
                    frame(st.pend, st.pend_pos)
                st.pend, st.pend_pos = np.zeros(0, bool), pos
            if st.pend_pos is None:
                st.pend_pos = pos
            st.pend = np.concatenate((st.pend, x))
            per = self.length - self.header_bits
            while len(st.pend) >= per:
                frame(st.pend[:per], st.pend_pos)
                st.pend, st.pend_pos = st.pend[per:], st.pend_pos + per
            if data.flush and len(st.pend):
                frame(st.pend, st.pend_pos)
                st.pend, st.pend_pos = np.zeros(0, bool), st.pend_pos + len(st.pend)
        y = np.concatenate(out) if out else np.zeros(0, bool)
        return data.derive(y, self.CATEGORY, dtype.BITS, Digital, ctx, rep=Rep.DATA)

    def __rmul__(self, data):
        expect(self, data, self.takes(True), True)
        return self.encode(data)

    def __rtruediv__(self, data):
        expect(self, data, self.takes(False), False)
        return self.decode(data)

    @staticmethod
    def load(name, obj):
        if "fields" not in obj:
            raise ConfigError(f"header {name}: needs 'fields'")
        return Header(name, obj["fields"], obj.get("order", "msb"), obj.get("crc"), obj.get("payload"), obj.get("preamble"),
                      obj.get("framed", False))

def _crc(name):
    from .serial import load
    return load("crc", name)._enc
