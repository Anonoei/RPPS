"""Link headers: fields, a CRC, and the payload they announce (rpps.profile.control)

A definition, config/header/link/<name>.json, `rp.link.load("header/ieee80211b_plcp")`:

    {"base": "header", "type": "link", "comment": "...", "ref": "...",
     "order": "lsb",                                      field bits sent least ("lsb") or most ("msb") significant first
     "payload_fields": [...],                             what the payload says, for its message (rpps.link.fields), read
                                                          with the header's fields in scope: Message.fields["payload"]
     "payload_order": "lsb",                              the payload's bytes sent least significant bit first (BLE, 802.11);
                                                          out (and in, transmitting) come bytes most significant bit first,
                                                          as everywhere else. Default "msb": the payload as sent
     "fields": [{"name": "signal", "bits": 8},               (a schema: rpps.link.fields; conditional, repeated and
                                                          expression-sized fields make a header read as it comes)
                {"name": "length", "bits": 16,
                 "tx": "(bits*10 + signal - 1)//signal"},   transmit: its value from the payload's bits and the other
                ...],                                         fields (an expression, as "expr" takes, worked out per frame);
                                                              else the controlled value, its "default", or 0
     "crc": "crc16_genibus",                              a crc definition over the field bits, sent after them
     "fcs": "crc24_ble",                                  a crc definition over header and payload, sent after the payload
     "payload": "length*signal//10",                      receive: payload bits after the header, from the fields
     "preamble": {"ones": 128, "word": "0xF3A0", "bits": 16},   transmit: sent before the header (in `order`)
     "framed": true}                                      frames of `length` bits (a controlled parameter: DVB-S2 Kbch),
                                                          the payload padded to fill them

Receiving, each frame starts where a profile applies a change (a delimiter before it publishes frame starts), or, framed,
`length` bits after the last. The header's fields, with "bits" (the payload's length), are published from the payload's
first bit for the payload's length, and {"frame_bits": header + payload} from the frame's first bit (rp.base.control.publish),
so a profile can drive the rate of the payload and the length of the frame. A header whose CRC fails is counted, and its
frame dropped; so is a frame whose FCS fails (`fcs_failed`). Out come the payloads. Every frame is also emitted as a
message (rpps.profile.message) from its first bit: its fields, its payload, and crc_ok (the FCS's verdict, else the
header CRC's, else None); a failed one too (crc_ok False, the fields as far as they were read). `select` ({field: value}) keeps only
the frames whose fields have those values: the others are used up, counted (`skipped`), and neither given nor published
(one of a DVB-S2 multistream's input streams by its ISI).
Transmitting, a frame starts at each change a profile applies: its fields are the controlled values; framed, its payload
is `length` minus the header, else everything until the next frame start (one frame per call). The fields are published
from the payload's first bit, and {"frame_bits": the frame's bits} from the frame's first bit after the preamble, so the
steps after the header can restart per frame (a whitening sequence, a sync marker) where a profile's controls say so.
"""
from types import SimpleNamespace

import numpy as np

from .. import _config
from ..base import Digital, dtype
from ..base.rpps import Pipe, FRAME, failed_of
from ..base.types import Category, Rep
from ..errors import ConfigError
from ..base.control import publish, anchor, emit

class Header(Pipe):
    CATEGORY = Category.LINK
    SCOPE, SCOPE_NEEDS = FRAME, ("length",) # A frame's own header and payload, when a control gives its length
    EMITS = True # Every frame, good or failed, is a message (rpps.profile.message)

    def __init__(self, name, fields, order="msb", crc=None, payload=None, preamble=None, framed=False, fcs=None,
                 payload_order="msb", payload_fields=None):
        from .fields import Schema
        self.name = name
        if payload_order not in ("msb", "lsb"):
            raise ConfigError(f"header {name}: payload_order is msb or lsb, not {payload_order!r}")
        self.payload_lsb = payload_order == "lsb"
        self.schema = Schema(fields, order, f"header {name}") # (rpps.link.fields: optional, repeated, sized by expression)
        self.fields = self.schema.fields
        self.payload_schema = None if not payload_fields else Schema(payload_fields, "msb", f"header {name} payload")
        self.order = order
        self.crc = None if crc is None else _crc(crc)
        self.fcs = None if fcs is None else _crc(fcs)
        self.fcs_bits = 0 if self.fcs is None else self.fcs.width
        self.payload = payload
        self.framed = bool(framed)
        self.crc_bits = self.crc.width if self.crc is not None else 0
        self.field_bits = self.schema.fixed_bits # None: the header's size follows its fields (read as they come)
        self.header_bits = None if self.field_bits is None else self.field_bits + self.crc_bits
        if framed and self.header_bits is None:
            raise ConfigError(f"header {name}: a framed header has fields of fixed size")
        self.preamble = np.zeros(0, dtype=bool) if not preamble else np.concatenate((
            np.ones(int(preamble.get("ones", 0)), dtype=bool),
            self._value_bits(preamble["word"], int(preamble["bits"])) if "word" in preamble else np.zeros(0, bool)))
        self.values = {f["name"]: f.get("default", 0) for f in self.fields}
        self.length = None
        self.select = None # {field: value}: the frames kept (see the module)
        self._start = True
        self.CONTROLS = {"frame": "live", "length": "live", **{f["name"]: "live" for f in self.fields}}

    def __str__(self):
        return f"Header:{self.name}"

    def takes(self, encode):
        return {Rep.DATA, Rep.PACKETS} if encode else {Rep.DATA}

    def pos_rate(self, decode):
        if decode or not self.framed or not self.length:
            return 1.0
        return self.length/(self.length - self.header_bits - self.fcs_bits)

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

    def kernel(self, params, how):
        """The batch decoder of a framed header of fixed size with a known frame length, its payload as sent, no FCS and no
        `select` (rpps.link.kernels.HeaderDecode); None otherwise (the frames go through the operator one at a time)"""
        if how != "/":
            return None
        if params:
            self.control(**params)
        if not (self.framed and self.length and self.header_bits is not None and self.fcs is None and not self.select
                and not self.payload_lsb):
            return None
        from .kernels import HeaderDecode
        return HeaderDecode(self, self.length)

    # ---- bits

    def _value_bits(self, v, n):
        v = int(v, 0) if isinstance(v, str) else int(v)
        idx = np.arange(n) if self.order == "lsb" else np.arange(n - 1, -1, -1)
        return ((v >> idx) & 1).astype(bool)

    def _value(self, bits):
        b = np.asarray(bits, dtype=np.int64)
        w = 1 << (np.arange(len(b)) if self.order == "lsb" else np.arange(len(b) - 1, -1, -1))
        return int(b @ w)

    def need(self, bits):
        """The header's bits, CRC included, as far as these first bits of it say (more once more of it is read)"""
        if self.header_bits is not None:
            return self.header_bits
        from .fields import Short
        try:
            return self.schema.parse(bits)[1] + self.crc_bits
        except Short as e:
            return e.args[0] + self.crc_bits

    def parse(self, bits):
        """(fields, CRC ok) of a header's bits"""
        bits = np.asarray(bits).astype(bool)
        out, i = self.schema.parse(bits)
        ok = True
        if self.crc is not None:
            ok = self.crc.from_bits(bits[i:i + self.crc.width]) == self.crc.compute_bits(bits[:i])
        return out, ok

    def payload_values(self, payload, fields=None):
        """What `payload_fields` reads of a payload (bytes most significant bit first), the header's `fields` in scope;
        None when it doesn't fit"""
        if self.payload_schema is None or payload is None:
            return None
        from .fields import Short
        try:
            return self.payload_schema.parse(payload, {k: v for k, v in (fields or {}).items() if not isinstance(v, (list, dict))})[0]
        except (Short, ValueError, ConfigError):
            return None

    def _emit(self, meta, fields, body, ok, at):
        vals = self.payload_values(body, fields) if ok is not False else None
        emit(meta, self, None, fields if vals is None else {**fields, "payload": vals}, body, ok, at)

    def build(self, values):
        """A header's bits for these field values"""
        bits = self.schema.build(values)
        if self.crc is not None:
            bits = np.concatenate((bits, self.crc.field(bits)))
        return bits

    def _bytes_order(self, bits):
        """The payload between the order it is sent in and bytes most significant bit first (its own inverse); a last
        part byte, which a byte-wise payload never has, stays as it is"""
        if not self.payload_lsb or not len(bits):
            return bits
        bits = np.asarray(bits).astype(bool)
        whole = len(bits)//8*8
        return np.concatenate((bits[:whole].reshape(-1, 8)[:, ::-1].reshape(-1), bits[whole:]))

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
            mode="header", held=np.zeros(0, bool), start=None, left=0, frame_left=None, pend=np.zeros(0, bool), pend_pos=None,
            body=np.zeros(0, bool), tail=np.zeros(0, bool), fields=None, part=np.zeros(0, bool))
        ctx[key] = st
        return st

    def decode(self, data):
        ctx = dict(data.ctx)
        st = self._state(ctx, False)
        x = data.bits() if hasattr(data, "bits") else np.asarray(data.data)
        x = np.asarray(x).astype(bool)
        pos0 = data.meta.pos
        est = data.meta.stream.est.setdefault(self.name, {"frames": 0, "crc_failed": 0, "last": None})
        est.setdefault("fec_failed", 0)
        est.setdefault("fcs_failed", 0)
        est.setdefault("skipped", 0)
        failed = failed_of(data) # What the decoders before could not decode: those frames are dropped
        out, n_out, i = [], 0, 0
        if self._start:
            self._start = False
            self._restart(st, pos0)
        while i < len(x):
            if self.framed and self.length and st.frame_left is not None and st.frame_left <= 0:
                self._restart(st, pos0 + i) # Framed: the next frame follows this one
            if st.mode == "header":
                need = self.need(st.held)
                if len(st.held) < need: # (a header whose fields size it: what is read so far says how much more)
                    take = min(need - len(st.held), len(x) - i)
                    st.held = np.concatenate((st.held, x[i:i + take]))
                    i += take
                    self._used(st, take)
                    if len(st.held) < need:
                        break
                    continue
                fields, ok = self.parse(st.held)
                hb = len(st.held)
                if not ok:
                    est["crc_failed"] += 1
                    emit(data.meta, self, None, fields, None, False, st.start)
                    st.mode = "skip"
                    continue
                n = self.payload_bits(fields)
                if self.framed and self.length:
                    n = min(n if n is not None else self.length, self.length - self.header_bits - self.fcs_bits)
                if failed is not None and failed[max(0, i - hb):i + (n or 0) + self.fcs_bits].any():
                    est["fec_failed"] += 1 # The code under it failed: its header passing its CRC proves nothing about the data
                    emit(data.meta, self, None, fields, None, False, st.start)
                    st.mode, st.left = "drop", (n or 0) + self.fcs_bits
                    continue
                if self.select and any(fields.get(k) != v for k, v in self.select.items()):
                    est["skipped"] += 1 # Not one of the frames kept
                    st.mode, st.left = "drop", (n or 0) + self.fcs_bits
                    continue
                if self.fcs is None:
                    est["frames"] += 1
                est["last"] = fields
                at = st.start + hb
                whole = hb + n + self.fcs_bits
                publish(data.meta, self, {**fields, "bits": n}, at=at, out=n_out, span=n, group=st.start)
                publish(data.meta, self, {"frame_bits": whole}, at=st.start, out=None, span=whole, group=st.start)
                anchor(data.meta, self, at, n_out)
                st.mode, st.left, st.body, st.fields, st.part = "payload", n, np.zeros(0, bool), fields, np.zeros(0, bool)
            elif st.mode == "payload":
                take = min(st.left, len(x) - i)
                if self.fcs is None: # (with an FCS: held until it says the frame is good)
                    part = np.concatenate((st.part, x[i:i + take])) if self.payload_lsb else x[i:i + take]
                    give = len(part)//8*8 if self.payload_lsb and st.left > take else len(part) # whole bytes, then the rest
                    out.append(self._bytes_order(part[:give]))
                    n_out += give
                    st.part = part[give:]
                st.body = np.concatenate((st.body, x[i:i + take]))
                i += take
                st.left -= take
                self._used(st, take)
                if st.left <= 0:
                    st.mode = "skip" if self.fcs is None else "fcs"
                    st.tail = np.zeros(0, bool)
                    if self.fcs is None:
                        body = self._bytes_order(st.body)
                        self._emit(data.meta, st.fields, body, True if self.crc is not None else None, st.start)
            elif st.mode == "fcs": # The frame check sequence over header and payload
                take = min(self.fcs_bits - len(st.tail), len(x) - i)
                st.tail = np.concatenate((st.tail, x[i:i + take]))
                i += take
                self._used(st, take)
                if len(st.tail) < self.fcs_bits:
                    break
                good = np.array_equal(self.fcs.field(np.concatenate((st.held, st.body))), st.tail)
                body = self._bytes_order(st.body) # (the FCS is over the bits as sent)
                if good:
                    out.append(body)
                    n_out += len(body)
                    est["frames"] += 1
                else:
                    est["fcs_failed"] += 1
                self._emit(data.meta, st.fields, body, bool(good), st.start)
                st.mode = "skip"
            elif st.mode == "drop": # A failed frame's payload: used up, not given out
                take = min(st.left, len(x) - i)
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
        ctx = dict(data.ctx)
        st = self._state(ctx, True)
        x = data.bits() if hasattr(data, "bits") else np.asarray(data.data)
        x = np.asarray(x).astype(bool).reshape(-1)
        pos = data.meta.pos
        out, n_out = [], 0
        def frame(payload, at):
            nonlocal n_out
            payload = self._bytes_order(payload) # As sent: the FCS is over these
            v = self._tx_values(len(payload))
            pad = np.zeros(0, bool)
            if self.framed and self.length:
                pad = np.zeros(self.length - self.header_bits - self.fcs_bits - len(payload), bool)
            hdr = self.build(v)
            head = np.concatenate((self.preamble, hdr))
            tail = np.zeros(0, bool) if self.fcs is None else self.fcs.field(np.concatenate((hdr, payload)))
            anchor(data.meta, self, at, n_out)
            publish(data.meta, self, {**v, "bits": len(payload)}, at=at, out=n_out + len(head), span=len(payload))
            whole = len(hdr) + len(payload) + len(tail) + len(pad)
            publish(data.meta, self, {"frame_bits": whole}, at=at, out=n_out + len(self.preamble), span=None, group=at) # (no end: the next frame start ends it)
            out.append(np.concatenate((head, payload, tail, pad)))
            n_out += len(head) + len(payload) + len(tail) + len(pad)
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
            per = self.length - self.header_bits - self.fcs_bits
            while len(st.pend) >= per:
                frame(st.pend[:per], st.pend_pos)
                st.pend, st.pend_pos = st.pend[per:], st.pend_pos + per
            if data.flush and len(st.pend):
                frame(st.pend, st.pend_pos)
                st.pend, st.pend_pos = np.zeros(0, bool), st.pend_pos + len(st.pend)
        y = np.concatenate(out) if out else np.zeros(0, bool)
        return data.derive(y, self.CATEGORY, dtype.BITS, Digital, ctx, rep=Rep.DATA)


    @staticmethod
    def load(name, obj):
        if "fields" not in obj:
            raise ConfigError(f"header {name}: needs 'fields'")
        return Header(name, obj["fields"], obj.get("order", "msb"), obj.get("crc"), obj.get("payload"), obj.get("preamble"),
                      obj.get("framed", False), obj.get("fcs"), obj.get("payload_order", "msb"), obj.get("payload_fields"))

def _crc(name):
    from .serial import load
    return load(f"crc/{name}").encoder
