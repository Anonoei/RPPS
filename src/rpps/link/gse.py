"""Generic Stream Encapsulation (ETSI TS 102 606-1): PDUs in the data fields of DVB-S2 BBFRAMEs

    {"link": "gse", "protocols": [130]}        receive: the PDUs of those protocol types, each a frame of its own

A data field holds GSE packets one after the other, then padding (a packet header of zeros) to its end; a packet never
crosses into the next data field, so a PDU too long for what is left is sent in fragments (Start, End indicators; a
fragment id), the first saying the PDU's total length, the last carrying a CRC-32 (MPEG-2) over the whole of it.

    S E LT GSE_Length(12) [Frag_ID(8)] [Total_Length(16)] [Protocol_Type(16) Label(6 | 3 | 0 bytes)] data [CRC_32]

Receiving takes the data fields as a frame each (its start where a profile's BBHEADER says one starts: the step's
`frame` control); every whole PDU is a message (kind "gse": protocol, label, length, its bytes), a fragmented one whose
CRC fails a failed message. The PDUs of the `protocols` asked for (all: none given) come out, each published as a frame
of its own ({"frame": k, "protocol", "pdu_bits"}) so the steps after (a table's header) read each by itself.
Protocol types below 0x0600 are taken as the PDU's own type, not an extension header (DVB-RCS2's NCR 0x0081 and lower
layer signalling 0x0082 are used so); extension headers are not read.

Transmitting a stream of data fields is `pack(pdus, size)`: the PDUs fragmented across data fields of `size` bytes,
each padded to its end (what a profile's BBHEADER then frames).
"""
import numpy as np

from ..base import Digital, dtype
from ..base.rpps import Pipe
from ..base.types import Category, Rep
from ..base.control import publish, anchor, emit
from ..errors import ConfigError

LABEL_BYTES = {0: 6, 1: 3, 2: 0, 3: 0} # Label type: 6 bytes, 3 bytes, none, re-use the last

def _crc32(data):
    from .crc import crc as _crc
    from ..logger import Logger
    global _CRC
    if _CRC is None:
        _CRC = _crc(Logger("gse"), 32, 0x04C11DB7, 0xFFFFFFFF, False, False, 0, 0x0376E6E7)
    return _CRC.compute(bytes(data))

_CRC = None

def pack(pdus, size, frag_ids=None):
    """Data fields of `size` bytes carrying the PDUs [(protocol type, payload bytes) or (protocol, payload, label
    bytes)] in GSE packets: whole where they fit, else fragmented (the first fragment as long as the field allows, the
    rest in the next fields), every field padded with zeros"""
    fields, cur = [], bytearray()
    def room():
        return size - len(cur)
    def close():
        nonlocal cur
        fields.append(bytes(cur) + bytes(size - len(cur)))
        cur = bytearray()
    fid = 0
    for item in pdus:
        proto, data = int(item[0]), bytes(item[1])
        label = bytes(item[2]) if len(item) > 2 and item[2] is not None else b""
        lt = {6: 0, 3: 1, 0: 2}.get(len(label))
        if lt is None:
            raise ValueError("gse.pack: a label is 6, 3 or 0 bytes")
        head = proto.to_bytes(2, "big") + label
        whole = 2 + len(head) + len(data)
        if whole > 4097:
            raise ValueError("gse.pack: a PDU over 4095 bytes needs a fragmenting packet of its own; not supported")
        if room() < 2 + len(head) + 1: # (no room for this packet's header and a byte: the next field)
            close()
        if whole <= room(): # Whole: S=1 E=1
            cur += (0xC000 | (lt << 12) | (len(head) + len(data))).to_bytes(2, "big") + head + data
            continue
        # Fragmented: total length over protocol, label and data; CRC-32 over total length, them and the data
        total = (len(head) + len(data)).to_bytes(2, "big")
        crc = _crc32(total + head + data).to_bytes(4, "big")
        body = head + data + crc # (the bytes the fragments carry after the first's frag id and total length)
        first = True
        frag = frag_ids[fid % len(frag_ids)] if frag_ids else fid % 256
        fid += 1
        while body:
            over = 2 + 1 + (2 if first else 0) # GSE header, frag id, total length (the first)
            if room() < over + 1:
                close()
            n = min(len(body), room() - over)
            last = n == len(body)
            s, e = (1 if first else 0), (1 if last else 0)
            if first and last: # (it fits after all: the packet is whole)
                raise AssertionError("gse.pack")
            ltv = lt if first else 3
            length = 1 + (2 if first else 0) + n
            cur += ((s << 15) | (e << 14) | (ltv << 12) | length).to_bytes(2, "big") + bytes([frag]) \
                   + (total if first else b"") + body[:n]
            body, first = body[n:], False
            if body:
                close()
    if cur or not fields:
        close()
    return fields


class GSE(Pipe):
    CATEGORY = Category.LINK
    CONTROLS = {"frame": "live"} # A data field starts here (a BBHEADER's): what is left of the last is padding
    EMITS = True
    check_name = "crc32_mpeg2" # A fragmented PDU's CRC-32 (whole PDUs have none: crc_ok then says it was reassembled)

    def __init__(self, protocols=None, name="gse"):
        self.protocols = None if protocols is None else {int(p) for p in protocols}
        self.name = name
        self._start = True

    def __str__(self):
        return f"GSE({sorted(self.protocols) if self.protocols else 'all'})"

    def takes(self, encode):
        return {Rep.DATA, Rep.PACKETS}

    def pos_rate(self, decode):
        return None # (PDUs of varying length, measured)

    def control(self, frame=None):
        self._start = True

    def encode(self, data):
        raise ConfigError("gse: transmitting is rpps.link.gse.pack (GSE packets fill each BBFRAME's data field)")

    def _state(self, ctx):
        key = (self, "rx")
        st = ctx.get(key)
        if st is None:
            st = ctx[key] = {"buf": np.zeros(0, bool), "pos": None, "frags": {}, "k": 0, "label": b""}
        return st

    def decode(self, data):
        ctx = dict(data.ctx)
        st = dict(self._state(ctx))
        ctx[(self, "rx")] = st
        x = np.asarray(data.bits()).astype(bool)
        pos0 = data.meta.pos
        est = data.meta.stream.est.setdefault(self.name, {"pdus": 0, "crc_failed": 0, "lost": 0})
        if self._start or st["pos"] is None: # A new data field: what the last left was padding
            self._start = False
            st["buf"], st["pos"] = x[:0], pos0
        st["buf"] = np.concatenate((st["buf"], x))
        out, n_out = [], 0
        while True:
            buf = st["buf"]
            if len(buf) < 16:
                break
            h = int(np.packbits(buf[:16]).view(">u2")[0])
            s, e, lt, n = h >> 15, (h >> 14) & 1, (h >> 12) & 3, h & 0xFFF
            if s == 0 and e == 0 and lt == 0: # Padding: the rest of the field
                st["buf"], st["pos"] = buf[:0], st["pos"] + len(buf)
                break
            if len(buf) < 16 + 8*n:
                break
            body = np.packbits(buf[16:16 + 8*n]).tobytes()
            at = st["pos"]
            st["buf"], st["pos"] = buf[16 + 8*n:], st["pos"] + 16 + 8*n
            pdu = self._packet(st, s, e, lt, body, est, data.meta, at)
            if pdu is None:
                continue
            proto, label, payload, start = pdu
            est["pdus"] += 1
            emit(data.meta, self, "gse", {"protocol": proto, "label": label.hex(), "length": len(payload)},
                 np.unpackbits(np.frombuffer(payload, np.uint8)).astype(bool) if payload else np.zeros(0, bool), True, start)
            if self.protocols is None or proto in self.protocols:
                bits = np.unpackbits(np.frombuffer(payload, np.uint8)).astype(bool)
                publish(data.meta, self, {"frame": st["k"], "protocol": proto, "pdu_bits": len(bits)}, at=start, out=n_out)
                anchor(data.meta, self, start, n_out)
                st["k"] += 1
                out.append(bits)
                n_out += len(bits)
        y = np.concatenate(out) if out else np.zeros(0, bool)
        return data.derive(y, self.CATEGORY, dtype.BITS, Digital, ctx, rep=Rep.DATA) # (PDUs: data a header after it reads)

    def _packet(self, st, s, e, lt, body, est, meta, at):
        """(protocol, label, payload, start position) of a whole PDU this packet ends, or None"""
        frags = st["frags"]
        if s and e: # Whole
            proto = int.from_bytes(body[:2], "big")
            nl = LABEL_BYTES[lt]
            label = body[2:2 + nl] if lt != 3 else st["label"]
            if lt in (0, 1):
                st["label"] = label
            return proto, label, body[2 + nl:], at
        fid = body[0]
        if s: # First fragment: total length, protocol, label, the start of the data
            frags[fid] = {"total": int.from_bytes(body[1:3], "big"), "data": bytearray(body[3:]), "at": at, "lt": lt}
            st["frags"] = dict(frags)
            return None
        f = frags.get(fid)
        if f is None: # A fragment whose start was not seen
            est["lost"] += 1
            return None
        f["data"] += body[1:]
        if not e:
            return None
        frags.pop(fid, None)
        data, crc = bytes(f["data"][:-4]), int.from_bytes(f["data"][-4:], "big")
        if len(data) != f["total"] or _crc32(f["total"].to_bytes(2, "big") + data) != crc:
            est["crc_failed"] += 1
            emit(meta, self, "gse", {"length": len(data)}, None, False, f["at"])
            return None
        nl = LABEL_BYTES[f["lt"]]
        proto, label = int.from_bytes(data[:2], "big"), data[2:2 + nl]
        return proto, label, data[2 + nl:], f["at"]
