"""Packets that cross frames: a stream of packets carried in the data fields of fixed frames, cut back into packets

    {"link": "packets", "id": "space", "params": ["fhp", "vcid", "vc_count"],          what a header before it gives
     "frame_fields": [...],                         read at the start of each frame's data (in band: a pointer byte)
     "active": "1",                                  frames that carry packets at all (others are passed over)
     "starts": "fhp != 2047",                        a packet starts in this frame (else it only continues one)
     "pointer": "8*fhp",                             where, in bits from the data's start (after the frame_fields)
     "idle_frame": "fhp == 2046",                    a frame of idle data only
     "channel": "vcid", "count": "vc_count", "modulo": 256,    streams kept apart, and their frame counts
     "fields": [...],                                the packet's header (rpps.link.fields)
     "length": "8*(length + 7)",                     its whole length in bits, from those fields
     "idle": "apid == 2047",                         idle packets: used up, not given
     "end": "table_id == 255",                       filler: the rest of the frame is not packets
     "carried_check": {"crc": "crc8_dvbs2", "bits": 8, "restore": "0x47"},
     "keep": "8*188",                                the packet's first bits given; the rest of it is its trailer
     "trailer": [...],                               fields read there and taken off (DVB-S2's ISSY and DNP)
     "insert": {"count": "dnp", "bits": 1504,         packets put back before it: `count` of them, each `head` then
                "head": "0x471FFF10", "fill": 255},   `fill` bytes to `bits` (the null packets DVB-S2 deleted)
     "defaults": {"upl": 1504},                      values of the params before any control (transmit: all there is)
     "tx": {"syncd": "pointer if start else 65535"}} transmit: the fields published at each frame's start

A header step before it (a transfer frame's, a BBHEADER, a transport packet's) drives it from each frame's data: the
profile's `controls` give it `frame` (a frame starts here), `length` (the data's bits; `bits` of a header) and the
`params` it names. Each frame's data is taken whole, then cut: what comes before the pointer finishes the packet the
last frame of its channel left open; from the pointer, packets follow one another, each `length` long, the last one
left open for the next frame. A frame lost (its channel's count skips: `modulo`), or a packet whose end doesn't meet
the pointer, drops only the packets that crossed it (`est[id]`: "lost", "misaligned"); the stream picks up again at
the next pointer. Bits left before a pointer that are fewer than a header are filler (no packet fits there). A
packet's length of zero or over `max` bits (default 65536*8) is taken as lost sync.

Every packet is a message (kind: the step's id) with its header's fields and the `channel`, its frame the whole
packet (parts: header, payload); good packets come out, one after another, each published as a frame of its own
({"frame", "packet_bits"}) for a step after it.

**Trailer and inserted packets** (DVB-S2 null-packet deletion, EN 302 307-1 annex D.3): a user packet `length` long
carries more than the packet it stands for (an ISSY field, the DNP byte). Once checked, its first `keep` bits are the
packet; the `trailer` fields are read from the rest (into its message's fields) and taken off. `insert` then puts
`count` packets back before it (the null packets deleted since the last one: `bits` long, `head` then `fill` bytes);
they are given and published as frames like the others, but are not messages (counted: `est[id]["inserted"]`).

**Carried check** (DVB-S2 TS mode, EN 302 307-1 5.1.4): a packet's first `bits` are a check over the packet before
(all of it but those first bits): each packet is given once the next has said whether it was good, its first bits
then set to `restore` (the sync byte 0x47 the check stood in for). The last packet of a stream is given unchecked.

**Transmitting** packs a stream of packets back to back into the frames' data fields. A frame starts at each control
(`length`: its data field's bits, the profile's framer giving it); at each start the `tx` fields are worked out and
published for the header step after it to build that frame with: over the params (`defaults`), `pointer` (bits from the
data field's start to the first packet starting in it), `start` (1 when one does, else 0) and `frame` (the frames
so far). A carried check is made (each packet's first `bits` the check of the one before; the first packet's zeros).
The packets' lengths come from their own header (`fields`, `length`); a frame start is published once the first
packet starting at or after it is seen, so the output from a frame start may wait up to a packet. Not transmitted: `frame_fields` (a pointer
in the data field), a trailer of any bits (ISSY, null-packet deletion), channels of their own; the step refuses them.

Shipped: `ccsds/tm_packets` (space packets out of TM transfer frames by their first header pointer), `dvbs2/ts`
(transport stream packets out of BBFRAMEs in TS mode by SYNCD and UPL, null packets deleted put back; transmitted too),
`dvbs2/mpe` and `dvbs2/ule` (sections and SNDUs out of the transport packets of one PID, by the pointer field). A
profile's capture rule then sends them on.
"""
import numpy as np

from .. import _config
from ..base import Digital, dtype
from ..base.rpps import Pipe
from ..base.types import Category, Rep
from ..base.control import publish, anchor, emit
from ..errors import ConfigError
from .fields import Schema, Short

KEYS = {"link", "id", "comment", "params", "frame_fields", "active", "starts", "pointer", "idle_frame", "channel", "count",
        "modulo", "fields", "length", "idle", "end", "max", "carried_check", "keep", "trailer", "insert", "defaults", "tx"}

class Packets(Pipe):
    """Packets cut from frames' data fields (see module)"""
    CATEGORY = Category.LINK
    EMITS = True

    def __init__(self, spec, name="packets"):
        spec = dict(spec)
        bad = set(spec) - KEYS
        if bad:
            raise ConfigError(f"link packets {name}: unknown keys {sorted(bad)}; it takes {sorted(KEYS)}")
        self.name = name
        self.params = list(spec.get("params", []))
        self.CONTROLS = {"frame": "live", "length": "live", **{p: "live" for p in self.params}}
        self.frame_fields = Schema(spec["frame_fields"], "msb", f"packets {name} frame_fields") if spec.get("frame_fields") else None
        self.active, self.starts = spec.get("active"), spec.get("starts", "1")
        if "pointer" not in spec or "length" not in spec:
            raise ConfigError(f"link packets {name}: needs \"pointer\" and \"length\"")
        self.pointer, self.length = spec["pointer"], spec["length"]
        self.idle_frame, self.channel, self.count = spec.get("idle_frame"), spec.get("channel"), spec.get("count")
        self.modulo = int(spec.get("modulo", 0)) or None
        self.fields = Schema(spec.get("fields", []), "msb", f"packets {name} fields")
        self.header_bits = self.fields.fixed_bits
        if self.header_bits is None:
            raise ConfigError(f"link packets {name}: a packet's header has fields of fixed size")
        self.idle, self.end = spec.get("idle"), spec.get("end")
        self.max = int(spec.get("max", 8*65536))
        cc = spec.get("carried_check")
        self.carried = None
        if cc:
            from .serial import load
            self.carried = (load(f"crc/{cc['crc']}").encoder, int(cc["bits"]),
                            None if cc.get("restore") is None else _config.evaluate(str(cc["restore"]), {}, name))
            self.check_name = cc["crc"]
        else:
            self.check_name = None
        self.keep = spec.get("keep")
        self.trailer = Schema(spec["trailer"], "msb", f"packets {name} trailer") if spec.get("trailer") else None
        ins = spec.get("insert")
        self.insert = None
        if ins:
            bad = set(ins) - {"count", "bits", "head", "fill"}
            if bad or "count" not in ins or "bits" not in ins:
                raise ConfigError(f"link packets {name}: insert takes count, bits, head and fill (needs count and bits), not {sorted(bad)}")
            nbits, head = int(ins["bits"]), str(ins.get("head", "0x"))
            hb = 4*(len(head) - 2) if head.lower().startswith("0x") else 0
            hv = int(head, 16) if hb else 0
            fill = int(ins.get("fill", 0))
            pkt = np.concatenate((((hv >> np.arange(hb - 1, -1, -1)) & 1).astype(bool),
                                  np.tile(((fill >> np.arange(7, -1, -1)) & 1).astype(bool), max(0, nbits - hb)//8 + 1)))[:nbits]
            self.insert = (ins["count"], pkt)
        defaults = dict(spec.get("defaults", {}))
        if set(defaults) - set(self.params):
            raise ConfigError(f"link packets {name}: defaults {sorted(set(defaults) - set(self.params))} are not params ({self.params})")
        self.defaults = {p: int(defaults.get(p, 0)) for p in self.params}
        self.tx = dict(spec.get("tx", {}))
        self.values = dict(self.defaults)
        self.frame_len = None
        self._start = False

    def __str__(self):
        return f"Packets({self.name})"

    def takes(self, encode):
        return {Rep.DATA, Rep.PACKETS}

    def pos_rate(self, decode):
        return None if decode else 1.0 # (receiving: packets of varying length, measured; transmitting: bits kept)

    def control(self, frame=None, length=None, **params):
        unknown = set(params) - set(self.params)
        if unknown:
            raise ConfigError(f"link packets {self.name}: no parameter {sorted(unknown)} (params: {self.params})")
        if length is not None:
            self.frame_len = int(length)
        self.values = {**self.values, **{k: v for k, v in params.items() if v is not None}}
        self._start = True

    # ---- transmit

    def encode(self, data):
        if self.frame_fields is not None or self.channel is not None:
            raise ConfigError(f"link packets {self.name}: transmitting frame_fields or channels is not built")
        ctx = dict(data.ctx)
        key = (self, "tx")
        st = ctx.get(key)
        st = dict(st) if st is not None else {"buf": np.zeros(0, bool), "buf_pos": None, "pk": np.zeros(0, bool),
                                               "pk_start": None, "pk_len": None, "starts": [], "frames": [], "k": 0,
                                               "check": None, "rep": None}
        ctx[key] = st
        x = np.asarray(data.bits()).astype(bool).reshape(-1)
        pos = data.meta.pos
        if st["buf_pos"] is None:
            st["buf_pos"], st["pk_start"] = pos, pos
            if self.carried is not None:
                st["check"] = np.zeros(self.carried[1], bool)
        if self._start:
            self._start = False
            st["frames"] = st["frames"] + [(pos, self.frame_len)]
        y = self._tx_walk(st, x, pos)
        st["buf"] = np.concatenate((st["buf"], y))
        # What can go out: up to the first frame start not yet known (the packet crossing it hasn't said where it ends)
        out_pos, n_emit, pubs, frames = st["buf_pos"], len(st["buf"]), [], []
        for P, L in st["frames"]:
            p = self._tx_pointer(st, P, data.flush)
            if p is False:
                n_emit = min(n_emit, P - out_pos)
                frames.append((P, L))
                continue
            if frames: # (in order: one after an unknown one waits too)
                frames.append((P, L))
                continue
            start = p is not None and (L is None or p < L)
            v = {**self.values, "pointer": p if start else 0, "start": int(start), "frame": st["k"]}
            pubs.append((P, {k: int(self._ev(e, v)) for k, e in self.tx.items()}))
            st["k"] += 1
        st["frames"] = frames
        st["starts"] = [s_ for s_ in st["starts"] if frames and s_ >= frames[0][0]]
        y, st["buf"], st["buf_pos"] = st["buf"][:n_emit], st["buf"][n_emit:], out_pos + n_emit
        for P, f in pubs:
            publish(data.meta, self, f, at=P, out=P - out_pos)
        return data.derive(y, self.CATEGORY, dtype.BITS, Digital, ctx, rep=Rep.DATA)

    def _tx_pointer(self, st, P, flush):
        """Bits from P to the first packet starting there or after: None, none does (the stream ended); False: not known
        yet (no packet's first bit seen after P)"""
        later = [s_ for s_ in st["starts"] if s_ >= P]
        if later:
            return later[0] - P
        return None if flush else False

    def _tx_walk(self, st, x, pos):
        """Follow the packets through x (their starts, lengths), making each one's carried check: x as sent"""
        y = x.copy()
        i = 0
        while i < len(x):
            if not len(st["pk"]) and st["rep"] is None and st["check"] is not None: # A packet's first bits: the check
                st["rep"], st["check"] = (st["check"], 0), None                     # of the one before
            if st["rep"] is not None:
                bits, k = st["rep"]
                n = min(len(bits) - k, len(x) - i)
                y[i:i + n] = bits[k:k + n]
                st["rep"] = None if k + n >= len(bits) else (bits, k + n)
            if st["pk_len"] is None and len(st["pk"]) >= self.header_bits:
                fields = self.fields.parse(st["pk"][:self.header_bits])[0] if self.header_bits else {}
                values = {**self.values, **{k: v for k, v in fields.items() if not isinstance(v, (list, dict))}}
                n = int(self._ev(self.length, values))
                if n <= 0 or n > self.max:
                    raise ConfigError(f"link packets {self.name}: a packet to send is {n} bits long by its header")
                if self.keep is not None and n > int(self._ev(self.keep, values)):
                    raise ConfigError(f"link packets {self.name}: transmitting a trailer is not built")
                st["pk_len"] = n
            want = st["pk_len"] if st["pk_len"] is not None else self.header_bits
            take = min(want - len(st["pk"]), len(x) - i)
            if not len(st["pk"]) and take:
                st["starts"] = st["starts"] + [st["pk_start"]]
            st["pk"] = np.concatenate((st["pk"], x[i:i + take]))
            i += take
            if st["pk_len"] is not None and len(st["pk"]) >= st["pk_len"]: # Whole: the next one starts here
                if self.carried is not None:
                    st["check"] = self.carried[0].field(st["pk"][self.carried[1]:])
                st["pk_start"] += st["pk_len"]
                st["pk"], st["pk_len"] = st["pk"][:0], None
        return y

    # ---- receive

    def _state(self, ctx):
        key = (self, "rx")
        st = ctx.get(key)
        st = dict(st) if st is not None else {"frame": None, "fbuf": np.zeros(0, bool), "fpos": 0, "chans": {}, "k": 0}
        ctx[key] = st
        return st

    def _ev(self, expr, values):
        return _config.evaluate(str(expr), values, f"packets {self.name}")

    def decode(self, data):
        ctx = dict(data.ctx)
        st = self._state(ctx)
        x = np.asarray(data.bits()).astype(bool)
        pos0 = data.meta.pos
        est = data.meta.stream.est.setdefault(self.name, {"packets": 0, "idle": 0, "lost": 0, "misaligned": 0, "failed": 0,
                                                          "frames": 0})
        out = []
        if self._start: # A frame's data starts here: the last one, short or not, is done
            self._start = False
            if st["frame"] is not None:
                self._frame(st, est, data.meta, out)
            st["frame"], st["fbuf"], st["fpos"] = (dict(self.values), self.frame_len), x[:0], pos0
        if st["frame"] is not None:
            st["fbuf"] = np.concatenate((st["fbuf"], x))
            n = st["frame"][1]
            if n is not None and len(st["fbuf"]) >= n: # Whole: cut it now (what comes after it isn't this frame's)
                st["fbuf"] = st["fbuf"][:n]
                self._frame(st, est, data.meta, out)
                st["frame"] = None
        if data.flush:
            if st["frame"] is not None:
                self._frame(st, est, data.meta, out)
                st["frame"] = None
            for ch in st["chans"].values(): # The last packet of a carried check: given unchecked
                self._give(ch, None, est, data.meta, out)
        n_out, ys = 0, []
        for at, bits, own in out:
            publish(data.meta, self, {"frame": st["k"], "packet_bits": len(bits)}, at=at, out=n_out)
            if own: # (an inserted packet has no place of its own in the input)
                anchor(data.meta, self, at, n_out)
            st["k"] += 1
            ys.append(bits)
            n_out += len(bits)
        y = np.concatenate(ys) if ys else np.zeros(0, bool)
        return data.derive(y, self.CATEGORY, dtype.BITS, Digital, ctx, rep=Rep.PACKETS)

    def _frame(self, st, est, meta, out):
        """One frame's data: finish the open packet of its channel, then cut from the pointer"""
        values, _ = st["frame"]
        bits, at = st["fbuf"], st["fpos"]
        values = dict(values)
        if self.active is not None and not self._ev(self.active, values):
            return
        est["frames"] += 1
        used = 0
        if self.frame_fields is not None:
            try:
                got, used = self.frame_fields.parse(bits, {k: v for k, v in values.items() if not isinstance(v, (list, dict))})
            except (Short, ValueError, ConfigError):
                return
            values.update({k: v for k, v in got.items() if not isinstance(v, (list, dict))})
        bits, at = bits[used:], at + used
        key = self._ev(self.channel, values) if self.channel is not None else 0
        ch = st["chans"].get(key)
        if ch is None:
            ch = st["chans"][key] = {"buf": np.zeros(0, bool), "at": 0, "synced": False, "count": None, "held": None,
                                     "key": key}
        if self.count is not None:
            c = int(self._ev(self.count, values))
            if ch["count"] is not None and (c - ch["count"]) % (self.modulo or (1 << 32)) != 1: # A frame of it was lost
                self._lose(ch, est, meta, out)
            ch["count"] = c
        if self.idle_frame is not None and self._ev(self.idle_frame, values):
            return
        if not self._ev(self.starts, values): # It only continues a packet
            if ch["synced"]:
                self._append(ch, bits, at)
                self._cut(ch, est, meta, out)
            return
        p = int(self._ev(self.pointer, values))
        if not 0 <= p <= len(bits):
            est["misaligned"] += 1
            self._lose(ch, est, meta, out)
            return
        if ch["synced"]:
            self._append(ch, bits[:p], at)
            self._cut(ch, est, meta, out)
            if len(ch["buf"]) >= max(self.header_bits, 1): # The open packet doesn't end where the pointer says the next starts
                est["misaligned"] += 1
                self._lose(ch, est, meta, out)
            # (fewer bits than a header: no packet, filler: the 0xFF byte at the end of a transport packet's payload)
        ch["buf"], ch["at"], ch["synced"] = bits[p:], at + p, True
        self._cut(ch, est, meta, out)

    def _append(self, ch, bits, at):
        if not len(ch["buf"]):
            ch["at"] = at
        ch["buf"] = np.concatenate((ch["buf"], bits))

    def _lose(self, ch, est, meta, out):
        """Sync lost on a channel: its open packet dropped (counted), a carried check's held packet given unchecked"""
        if ch["synced"] and len(ch["buf"]):
            est["lost"] += 1
        ch["buf"], ch["synced"] = ch["buf"][:0], False
        self._give(ch, None, est, meta, out)

    def _cut(self, ch, est, meta, out):
        while ch["synced"] and len(ch["buf"]) >= self.header_bits:
            buf = ch["buf"]
            try:
                fields = self.fields.parse(buf[:self.header_bits])[0] if self.header_bits else {}
            except (Short, ValueError, ConfigError):
                fields = {}
            values = {**self.values, **{k: v for k, v in fields.items() if not isinstance(v, (list, dict))}}
            if self.end is not None and self.header_bits and self._ev(self.end, values): # Filler to the frame's end
                ch["buf"], ch["synced"] = buf[:0], False
                return
            n = int(self._ev(self.length, values))
            if n <= 0 or n > self.max:
                est["lost"] += 1
                ch["buf"], ch["synced"] = buf[:0], False
                return
            if len(buf) < n:
                return
            pkt, at = buf[:n], ch["at"]
            ch["buf"], ch["at"] = buf[n:], at + n
            if self.idle is not None and self.header_bits and self._ev(self.idle, values):
                est["idle"] += 1
                continue
            self._packet(ch, pkt, fields, at, est, meta, out)

    def _packet(self, ch, pkt, fields, at, est, meta, out):
        if self.carried is None:
            self._emit(ch, pkt, fields, at, None, est, meta, out)
            return
        crc, nb, _ = self.carried
        self._give(ch, pkt[:nb], est, meta, out) # Its first bits say whether the one before was good
        ch["held"] = (pkt, fields, at)

    def _give(self, ch, check_bits, est, meta, out):
        """A carried check's held packet, given now: checked by the next packet's first bits (None: unchecked)"""
        held = ch.get("held")
        if held is None:
            return
        ch["held"] = None
        pkt, fields, at = held
        crc, nb, restore = self.carried
        ok = None
        if check_bits is not None:
            ok = bool(np.array_equal(crc.field(pkt[nb:]), check_bits))
        if restore is not None:
            pkt = np.concatenate((((int(restore) >> np.arange(nb - 1, -1, -1)) & 1).astype(bool), pkt[nb:]))
        if ok is False:
            est["failed"] += 1
        self._emit(ch, pkt, fields, at, ok, est, meta, out, give=ok is not False)

    def _emit(self, ch, pkt, fields, at, ok, est, meta, out, give=True):
        fields = dict(fields)
        values = {**self.values, **{k: v for k, v in fields.items() if not isinstance(v, (list, dict))}}
        if self.keep is not None: # The packet, and its trailer read and taken off
            kb = int(self._ev(self.keep, values))
            tail, pkt = pkt[kb:], pkt[:kb]
            if self.trailer is not None:
                try:
                    got = self.trailer.parse(tail, values)[0]
                except (Short, ValueError, ConfigError):
                    got = {}
                fields.update(got)
                values.update({k: v for k, v in got.items() if not isinstance(v, (list, dict))})
        hb = self.header_bits
        f = {**fields, **({"channel": ch["key"]} if self.channel is not None else {})}
        emit(meta, self, None, f, pkt, ok, at, frame=pkt, parts={"header": (0, hb), "payload": (hb, len(pkt) - hb)} if hb
             else {"payload": (0, len(pkt))})
        if give:
            if self.insert is not None: # What was taken out before it, put back
                try:
                    n = int(self._ev(self.insert[0], values))
                except ConfigError:
                    n = 0
                est["inserted"] = est.get("inserted", 0) + n
                out += [(at, self.insert[1], False)]*n
            est["packets"] += 1
            out.append((at, pkt, True))
