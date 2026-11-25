"""Captures: decoded frames as packets of a link type, for Wireshark (rpps.source.PcapSink) or an interface (rpps.source.Tun)

A profile's messages (rpps.profile.message) hold what each step decoded; a capture says how a kind of message becomes
the bytes a link type expects, and back again on transmit. Profiles say it in `"capture"`, one rule a message kind:

    "capture": {"pdu": {"linktype": "ble_ll_phdr",                       a config/linktype definition (or "find", below)
                        "include": ["header", "payload", "fcs"],          the message's parts, in order (Message.parts)
                        "transform": [{"prefix": "0x8E89BED6", "bytes": 4, "order": "little"},
                                      {"header": "ble_phdr", "channel": "channel"}],
                        "drop": "type == 3",                               messages left out (an expression over fields)
                        "fields": ["type.name", "length"]},               fields named in the packet's comment (true: all)
                "frame": false}                                           a kind not captured

A kind without a rule goes out as Wireshark's exported PDU (`upper_pdu`) to its "data" dissector, so it is still seen.

**Parts.** A step that knows more of its frame than the payload emits the whole frame with where each part is (a link
header: header, payload, FCS; an HDLC frame: payload, FCS). `include` takes them in order; without it the payload alone.

**Transforms**, applied in order, each with an inverse for transmit (`unpacket`):

| Transform | Receive | Transmit |
|---|---|---|
| `{"bit_order": "lsb"}` | each byte's bits reversed | the same |
| `{"prefix": v, "bytes": n, "order": "big"}` | n bytes of `v` (a constant, a field, an expression) before | taken off (a field name: its value read back) |
| `{"suffix": v, "bytes": n, "order": "big"}` | after | taken off |
| `{"strip": {"head": n, "tail": m}, "fill": 0}` | n bytes off the front, m off the end (expressions) | put back, `fill` |
| `{"set": {"at": i, "value": v}, "was": w}` | byte i set to v (a restored sync byte) | set to w, when given |
| `{"fcs": "<crc definition>", "order": "big"}` | a check over the bytes so far, appended | checked and taken off |
| `{"header": "radiotap" \\| "ble_phdr", ...}` | a pseudo-header built from the message (below) | taken off |
| `{"restore": "ppp_acfc_pfc"}` | a named repair an expression can't make (below) | its inverse |

Expressions see the message's fields (a group's as `group.field`), the profile's parameters, and `size` (the bytes so far).

**Pseudo-headers.** `radiotap` (802.11): flags (FCS at the end when `fcs` is true), `rate` (an expression in 500 kb/s
units), `channel` (MHz; the message's frequency when not given), 2.4 GHz CCK. `ble_phdr` (LINKTYPE_BLUETOOTH_LE_LL_WITH_PHDR):
the RF `channel` (0..39), dewhitened, the CRC checked and its verdict (the message's crc_ok).

**Restores.** `ppp_acfc_pfc`: puts back the address and control bytes (0xFF 0x03) a peer left out (ACFC), and widens a
one-byte protocol field (PFC) to two (RFC 1661 6.5, 6.6); transmitting, nothing to undo (uncompressed is always valid).

**Finding a link type in a frame** (`"linktype": "find"`, `"find": {"in": ["ethernet", "ipv4", "ipv6"], "max": 64}`):
a frame with a header nobody describes, then a frame of a known link type. Each offset from `min` to `max` is tried
against the link types' `signature` (fields, a `valid` expression over them and `size`, a check over `bytes` that comes
out zero, and `next`: what follows by a field, an EtherType's IPv4); the first offset with the best score wins (1: the
fields are valid; 2: and the check holds; one more for each layer inside that scores too), at least `score`. The bytes before it are left out, and the packet takes that link type. Only where a frame starts is found:
nothing above the link layer is read. No match: the frame goes out whole, as exported PDU data, and is counted.

**Exported PDU** (`upper_pdu`): tags then the payload. The rule's `dissector` names Wireshark's dissector
(`"lte_rrc.bcch_dl_sch"`); the protocol column says the profile and kind, the info column the check and fields.
"""
import numpy as np

from .. import _config
from ..errors import ConfigError

# ---- link types

class LinkType:
    """A link type (config/linktype): its pcap number (`dlt`), the Linux ARPHRD a TUN takes for it (`arphrd`, `proto`:
    the EtherType a TUN's packet information says), Wireshark's dissector, what it says of an FCS, and a signature"""
    def __init__(self, name, dlt, arphrd=None, proto=None, dissector=None, fcs=None, signature=None, **kw):
        self.name, self.dlt = name, int(dlt)
        self.made_from = {k: list(v) for k, v in (kw.pop("from", None) or {}).items()} # {other link type: transforms}
        if kw:
            raise ConfigError(f"linktype {name}: unknown keys {sorted(kw)}")
        self.arphrd = None if arphrd is None else int(arphrd)
        self.proto = None if proto is None else (int(proto, 0) if isinstance(proto, str) else int(proto))
        self.dissector, self.fcs = dissector, fcs
        self.signature = self._plain = None
        if signature:
            from .fields import Schema
            sig = dict(signature)
            check = sig.get("check")
            nxt = sig.get("next")
            self._plain = None # Plain unsigned fields of fixed size: read as integers (fast), else through the schema
            if all(set(f) <= {"name", "bits", "comment"} and isinstance(f.get("bits"), int) for f in sig["fields"]):
                at, plain = 0, []
                for fl in sig["fields"]:
                    plain.append((fl["name"], at, fl["bits"]))
                    at += fl["bits"]
                self._plain = (plain, (at + 7)//8)
            self.signature = (Schema(sig["fields"], "msb", f"linktype {name} signature"), sig["valid"], int(sig.get("min", 1)),
                              None if check is None else (_check(check["crc"]), check.get("bytes")),
                              None if nxt is None else (nxt["field"], nxt["at"], {int(k, 0): v for k, v in nxt["linktypes"].items()}))

    def __repr__(self):
        return f"<LinkType {self.name} {self.dlt}>"

    def score(self, data, bits=None):
        """How well `data` starts a frame of this link type: 0 not at all, 1 its fields are valid, 2 and its check holds;
        with a `next` (what follows, by a field: an EtherType), one more than what follows scores, when it scores.
        bits: `data` unpacked already (MSB first), at least as far as the signature reads"""
        if self.signature is None:
            return 0
        schema, valid, least, check, nxt = self.signature
        if len(data) < least:
            return 0
        if self._plain is not None:
            plain, nbytes = self._plain
            if len(data) < nbytes:
                return 0
            word, total = int.from_bytes(bytes(data[:nbytes]), "big"), 8*nbytes
            values = {name: (word >> (total - at - n)) & ((1 << n) - 1) for name, at, n in plain}
            values["size"] = len(data)
        else:
            from .fields import Short
            need = schema.fixed_bits or 8*64
            if bits is None:
                bits = np.unpackbits(np.frombuffer(bytes(data[:(need + 7)//8]), np.uint8)).astype(bool)
            try:
                fields = schema.parse(bits[:need])[0]
            except (Short, ValueError, ConfigError):
                return 0
            values = {**_flat(fields), "size": len(data)}
        if not _config.evaluate(valid, values, f"linktype {self.name}"):
            return 0
        if nxt is not None:
            field, at, kinds = nxt
            inner = kinds.get(values.get(field))
            s = linktype(inner).score(memoryview(data)[int(at):], None if bits is None else bits[8*int(at):]) if inner else 0
            if s:
                return 1 + s
        if check is None:
            return 1
        crc, n = check
        n = len(data) if n is None else int(_config.evaluate(n, values, f"linktype {self.name}"))
        return 2 if 0 < n <= len(data) and crc.compute(bytes(data[:n])) == 0 else 1

    @staticmethod
    def load(name, obj):
        keys = ("dlt", "arphrd", "proto", "dissector", "fcs", "signature", "from")
        return LinkType(name.split("/")[-1], **{k: obj[k] for k in keys if k in obj})

DOMAIN = _config.register_domain("linktype", {"linktype": LinkType.load})
_LINKTYPES = {}

def linktype(name):
    """A link type by name: `linktype("ieee802_15_4")` (config/linktype)"""
    if isinstance(name, LinkType):
        return name
    lt = _LINKTYPES.get(name)
    if lt is None:
        lt = _LINKTYPES[name] = _config.load("linktype", name)
    return lt

def convert(data, src, dst):
    """A packet of link type `src` as one of link type `dst`, where one is the other with transforms (a link type's
    "from": {other: [transforms]}, as a capture rule's: 802.15.4 with FCS is 802.15.4 without, its FCS appended; AX.25
    with its KISS byte is AX.25 with a 0 before). ValueError where neither says how (or a check to take off fails)"""
    src, dst = linktype(src), linktype(dst)
    if src.name == dst.name:
        return bytes(data)
    if src.name in dst.made_from:
        return Capture({"linktype": dst.name, "transform": dst.made_from[src.name]}).forward(data)
    if dst.name in src.made_from:
        return Capture({"linktype": src.name, "transform": src.made_from[dst.name]}).unpacket(data)[0]
    raise ValueError(f"no conversion between link types {src.name} and {dst.name} (a link type's \"from\" says one)")

def by_dlt(dlt):
    """The link type of a pcap link type number (None: not one of config/linktype)"""
    return next((linktype(n) for n in linktypes() if linktype(n).dlt == int(dlt)), None)

def linktypes():
    """Every link type's name"""
    return sorted(n for ns in _config.ls("linktype").values() for n in ns)

def find(data, names, lo=0, hi=64, least=1):
    """(offset, LinkType) where a frame of one of these link types starts in `data`: the best score, the first offset
    of it, from `lo` to `hi`; (None, None) when none scores `least`"""
    kinds = [linktype(n) for n in names]
    data = bytes(data)
    allbits = None
    if any(lt._plain is None and lt.signature is not None for lt in kinds): # (unpacked once, for the schemas)
        allbits = np.unpackbits(np.frombuffer(data[:int(hi) + 128], np.uint8)).astype(bool)
    best = (0, None, None)
    view = memoryview(data)
    for off in range(int(lo), min(int(hi), len(data) - 1) + 1):
        for lt in kinds:
            s = lt.score(view[off:], None if allbits is None else allbits[8*off:])
            if s > best[0]:
                best = (s, off, lt)
    return (best[1], best[2]) if best[0] >= least else (None, None)

def _check(name):
    from .serial import load
    return load(f"crc/{name}").encoder

# ---- packets

class Packet:
    """One captured frame: its link type and bytes, and what it was (from its message)

    linktype: a LinkType; data: the bytes as the link type has them. profile, kind, check, crc_ok, sample, time, freq,
    quality: the message's; fields: the fields the rule names; comment(): one line of them, for a capture's comment"""
    __slots__ = ("linktype", "data", "profile", "kind", "check", "crc_ok", "sample", "time", "freq", "quality", "fields")

    def __init__(self, linktype, data, profile="", kind="", check=None, crc_ok=None, sample=0, time=None, freq=None,
                 quality=None, fields=None):
        self.linktype, self.data = linktype, bytes(data)
        self.profile, self.kind, self.check, self.crc_ok = profile, kind, check, crc_ok
        self.sample, self.time, self.freq, self.quality = sample, time, freq, quality
        self.fields = dict(fields or {})

    def comment(self):
        ok = {True: "ok", False: "FAILED", None: "unchecked"}[self.crc_ok]
        parts = [f"{self.profile} {self.kind}", f"{ok}" + (f" ({self.check})" if self.check else "")]
        if self.freq:
            parts.append(f"freq={self.freq/1e6:.6f} MHz")
        q = _quality(self.quality)
        if q is not None:
            parts.append(f"q={q:.1f} dB")
        parts.append(f"sample={self.sample}")
        parts += [f"{k}={_text(v)}" for k, v in self.fields.items()]
        return ", ".join(parts)

    def __repr__(self):
        return f"<Packet {self.linktype.name} {len(self.data)} bytes: {self.comment()}>"

def _quality(q):
    if q is None:
        return None
    if isinstance(q, (int, float)):
        return float(q)
    for k in ("esn0_db", "snr_db", "esn0", "snr"):
        v = getattr(q, k, None) if not isinstance(q, dict) else q.get(k)
        if isinstance(v, (int, float)):
            return float(v)
    return None

def _text(v):
    if isinstance(v, (bytes, bytearray)):
        return bytes(v).hex()
    if isinstance(v, np.generic):
        return str(v.item())
    if isinstance(v, float):
        return f"{v:g}"
    return str(v)

def _flat(fields, pre=""):
    """Scalar fields, a group's as group.field and a list's items as list.i(.field), for expressions and comments"""
    out = {}
    items = fields.items() if isinstance(fields, dict) else enumerate(fields or ())
    for k, v in items:
        if isinstance(v, (dict, list, tuple)):
            out.update(_flat(v, f"{pre}{k}."))
        elif isinstance(v, (int, float, bool, str, bytes, np.integer, np.floating)):
            out[f"{pre}{k}"] = v.item() if isinstance(v, np.generic) else v
    return out

def _value(v, values, where):
    """A transform's value: a number, a hex string, a field or parameter, or an expression over them"""
    if isinstance(v, (int, float)):
        return int(v)
    if isinstance(v, str) and v in values:
        return values[v]
    return _config.evaluate(str(v), values, where)

def _int_bytes(v, n, order):
    return (int(v) & ((1 << 8*n) - 1)).to_bytes(n, order)

def _rev(data):
    return bytes(_REV[np.frombuffer(bytes(data), np.uint8)])

_REV = np.array([int(f"{i:08b}"[::-1], 2) for i in range(256)], np.uint8)

# ---- pseudo-headers

def _radiotap(spec, msg, values, where):
    """A radiotap header: flags (FCS at the end), rate (500 kb/s units), channel (MHz, 2.4 GHz CCK)"""
    fcs = bool(_value(spec.get("fcs", 1), values, where))
    rate = int(_value(spec["rate"], values, where)) if "rate" in spec else None
    mhz = int(_value(spec["channel"], values, where)) if "channel" in spec else (int(round(msg.freq/1e6)) if msg.freq else 0)
    present, body = 1 << 1, bytes([0x10 if fcs else 0]) # flags
    if rate is not None:
        present |= 1 << 2
        body += bytes([rate & 0xFF])
    else:
        body += b"\x00" # (pad: the channel field is 2-aligned)
    present |= 1 << 3
    body += mhz.to_bytes(2, "little") + (0x0080 | 0x0020).to_bytes(2, "little") # 2 GHz spectrum, CCK
    return bytes([0, 0]) + (8 + len(body)).to_bytes(2, "little") + present.to_bytes(4, "little") + body

def _radiotap_len(data):
    return int.from_bytes(data[2:4], "little") if len(data) >= 4 else len(data)

def _ble_phdr(spec, msg, values, where):
    """LINKTYPE_BLUETOOTH_LE_LL_WITH_PHDR's pseudo-header: RF channel, signal and noise (not given), the reference access
    address (not given), flags: dewhitened, CRC checked, CRC valid"""
    channel = int(_value(spec.get("channel", 0), values, where))
    rf = {37: 0, 38: 12, 39: 39}.get(channel, channel + 1 + (channel > 10)) if spec.get("index", True) else channel
    flags = 0x0001 # dewhitened
    if msg.crc_ok is not None:
        flags |= 0x0400 | (0x0800 if msg.crc_ok else 0)
    return bytes([rf & 0xFF, 0, 0, 0]) + bytes(4) + flags.to_bytes(2, "little")

HEADERS = {"radiotap": (_radiotap, _radiotap_len), "ble_phdr": (_ble_phdr, lambda data: 10)}

# ---- restores

def _ppp_acfc_pfc(data):
    out = bytes(data)
    if out[:2] != b"\xff\x03":
        out = b"\xff\x03" + out
    if len(out) > 2 and out[2] & 1: # A protocol field's first byte is even: an odd one is a compressed (one-byte) field
        out = out[:2] + b"\x00" + out[2:]
    return out

RESTORES = {"ppp_acfc_pfc": (_ppp_acfc_pfc, lambda data: bytes(data))}

# ---- exported PDU

EXP_PDU_TAG_DISSECTOR_NAME, EXP_PDU_TAG_COL_PROT_TEXT, EXP_PDU_TAG_COL_INFO_TEXT = 12, 33, 36

def exported_pdu(payload, dissector, protocol=None, info=None):
    """Wireshark's exported PDU (LINKTYPE_WIRESHARK_UPPER_PDU): tags (dissector, protocol and info columns), the end tag,
    then the payload. Each tag: its type and length (16 bits, big endian), the value padded with zeros to 4 bytes"""
    def tag(t, text):
        v = text.encode()
        v += bytes(-len(v) % 4)
        return t.to_bytes(2, "big") + len(v).to_bytes(2, "big") + v
    out = tag(EXP_PDU_TAG_DISSECTOR_NAME, dissector)
    if protocol:
        out += tag(EXP_PDU_TAG_COL_PROT_TEXT, protocol)
    if info:
        out += tag(EXP_PDU_TAG_COL_INFO_TEXT, info)
    return out + bytes(4) + bytes(payload)

def unexported_pdu(data):
    """(dissector, payload) of an exported PDU"""
    i, name = 0, None
    while i + 4 <= len(data):
        t, n = int.from_bytes(data[i:i + 2], "big"), int.from_bytes(data[i + 2:i + 4], "big")
        if t == 0:
            return name, bytes(data[i + 4:])
        if t == EXP_PDU_TAG_DISSECTOR_NAME:
            name = bytes(data[i + 4:i + 4 + n]).rstrip(b"\x00").decode()
        i += 4 + n
    raise ValueError("exported PDU: no end tag")

# ---- one kind's rule

TRANSFORMS = ("bit_order", "prefix", "suffix", "strip", "set", "fcs", "header", "restore")
RULE_KEYS = ("linktype", "include", "transform", "drop", "fields", "dissector", "find", "comment")
PARTS = ("preamble", "header", "payload", "fcs", "trailer")

class Capture:
    """One message kind's capture rule (see module): `packet(message)` the Packet (None: dropped), `unpacket(bytes)` the
    payload bytes and fields to transmit"""
    def __init__(self, spec, where="capture", params=None):
        spec = dict(spec)
        bad = set(spec) - set(RULE_KEYS)
        if bad:
            raise ConfigError(f"{where}: unknown keys {sorted(bad)}; a capture rule takes {list(RULE_KEYS)}")
        self.where, self.params = where, dict(params or {})
        name = spec.get("linktype", "upper_pdu")
        self.find = None
        if name == "find":
            f = dict(spec.get("find") or {})
            if not f.get("in"):
                raise ConfigError(f"{where}: \"linktype\": \"find\" needs \"find\": {{\"in\": [link types], \"max\": bytes}}")
            self.find = ([linktype(n) for n in f["in"]], int(f.get("min", 0)), int(f.get("max", 64)), int(f.get("score", 1)))
            self.linktype = None
        else:
            self.linktype = linktype(name)
        self.dissector = spec.get("dissector") or (self.linktype.dissector if self.linktype is not None else None)
        if self.linktype is not None and self.linktype.name == "upper_pdu" and not spec.get("dissector"):
            self.dissector = "data"
        self.include = list(spec.get("include") or ["payload"])
        bad = [p for p in self.include if p not in PARTS]
        if bad:
            raise ConfigError(f"{where}: include takes {list(PARTS)}, not {bad}")
        self.transform = [dict(t) for t in spec.get("transform", [])]
        for t in self.transform:
            kind = [k for k in t if k in TRANSFORMS]
            if len(kind) != 1:
                raise ConfigError(f"{where}: a transform is one of {list(TRANSFORMS)}, got {t}")
            if kind[0] == "header" and t["header"] not in HEADERS:
                raise ConfigError(f"{where}: no pseudo-header {t['header']!r}; have {sorted(HEADERS)}")
            if kind[0] == "restore" and t["restore"] not in RESTORES:
                raise ConfigError(f"{where}: no restore {t['restore']!r}; have {sorted(RESTORES)}")
            if kind[0] == "fcs":
                t["_crc"] = _check(t["fcs"])
        self.drop = spec.get("drop")
        self.fields = spec.get("fields", True)
        self.undetected = 0 # Frames "find" found no link type in

    # -- receive

    def _bytes(self, msg):
        """The included parts of the message's frame, as bytes"""
        frame, nbits, parts = msg.whole()
        bits = np.unpackbits(np.frombuffer(frame, np.uint8))[:nbits]
        got = [bits[parts[p][0]:parts[p][0] + parts[p][1]] for p in self.include if p in parts]
        b = np.concatenate(got) if got else np.zeros(0, np.uint8)
        return np.packbits(b).tobytes()

    def values(self, msg, size=0):
        return {**self.params, **_flat(msg.fields), "crc_ok": msg.crc_ok, "size": size}

    def packet(self, msg):
        """The message as a Packet of this rule's link type; None when `drop` says so"""
        values = self.values(msg)
        if self.drop is not None and _config.evaluate(self.drop, values, self.where):
            return None
        data = self._bytes(msg)
        for t in self.transform:
            values["size"] = len(data)
            data = self._forward(t, data, msg, values)
        lt, dissector = self.linktype, self.dissector
        if self.find is not None:
            kinds, lo, hi, least = self.find
            off, lt = find(data, [k.name for k in kinds], lo, hi, least)
            if lt is None:
                self.undetected += 1
                lt, dissector = linktype("upper_pdu"), "data"
            else:
                data = data[off:]
        p = Packet(lt, data, msg.profile, msg.kind, msg.check, msg.crc_ok, msg.sample, msg.time, msg.freq, msg.quality,
                   self._fields(msg))
        if lt.name == "upper_pdu":
            p.data = exported_pdu(data, dissector or "data", f"{msg.profile} {msg.kind}", _info(p))
        return p

    def _fields(self, msg):
        if not self.fields:
            return {}
        flat = _flat(msg.fields)
        if self.fields is True:
            return {k: v for k, v in flat.items() if k != "bytes"}
        return {k: flat[k] for k in self.fields if k in flat}

    def _forward(self, t, data, msg, values):
        w = self.where
        if "bit_order" in t:
            return _rev(data) if t["bit_order"] == "lsb" else data
        if "prefix" in t or "suffix" in t:
            v = _int_bytes(_value(t.get("prefix", t.get("suffix")), values, w), int(t["bytes"]), t.get("order", "big"))
            return v + data if "prefix" in t else data + v
        if "strip" in t:
            s = t["strip"]
            head = int(_value(s.get("head", 0), values, w))
            tail = int(_value(s.get("tail", 0), values, w))
            return data[head:len(data) - tail]
        if "set" in t:
            s = t["set"]
            i = int(_value(s["at"], values, w))
            if not 0 <= i < len(data):
                return data
            return data[:i] + bytes([int(_value(s["value"], values, w)) & 0xFF]) + data[i + 1:]
        if "fcs" in t:
            crc = t["_crc"]
            return data + _int_bytes(crc.compute(data), crc.width//8, t.get("order", "big"))
        if "header" in t:
            return HEADERS[t["header"]][0](t, msg, values, w) + data
        if "restore" in t:
            return RESTORES[t["restore"]][0](data)
        raise ConfigError(f"{w}: transform {t}")

    def forward(self, data):
        """The transforms applied to bytes alone (no message: no pseudo-headers, fields or parameters)"""
        data = bytes(data)
        for t in self.transform:
            data = self._forward(t, data, None, {**self.params, "size": len(data)})
        return data

    # -- transmit

    def unpacket(self, data):
        """(bytes, fields) for transmitting a packet of this rule's link type: the transforms undone in reverse (a prefix
        named by a field gives that field). The bytes are the included parts, in order (the profile's Encoder splits them)"""
        data, fields = bytes(data), {}
        if self.linktype is not None and self.linktype.name == "upper_pdu":
            data = unexported_pdu(data)[1]
        values = dict(self.params)
        for t in reversed(self.transform):
            data = self._inverse(t, data, values, fields)
        return data, fields

    def _inverse(self, t, data, values, fields):
        w = self.where
        if "bit_order" in t:
            return _rev(data) if t["bit_order"] == "lsb" else data
        if "prefix" in t or "suffix" in t:
            n = int(t["bytes"])
            v, rest = (data[:n], data[n:]) if "prefix" in t else (data[-n:], data[:-n])
            src = t.get("prefix", t.get("suffix"))
            if isinstance(src, str) and src.isidentifier():
                fields[src] = int.from_bytes(v, t.get("order", "big"))
            return rest
        if "strip" in t:
            s = t["strip"]
            values = {**values, **fields, "size": len(data)}
            head = int(_value(s.get("head", 0), values, w))
            tail = int(_value(s.get("tail", 0), values, w))
            fill = int(_value(t.get("fill", 0), values, w)) & 0xFF
            return bytes([fill])*head + data + bytes([fill])*tail
        if "set" in t:
            if "was" not in t:
                return data
            i = int(_value(t["set"]["at"], {**values, **fields}, w))
            if not 0 <= i < len(data):
                return data
            return data[:i] + bytes([int(_value(t["was"], values, w)) & 0xFF]) + data[i + 1:]
        if "fcs" in t:
            crc = t["_crc"]
            n = crc.width//8
            body, got = data[:-n], data[-n:]
            if _int_bytes(crc.compute(body), n, t.get("order", "big")) != got:
                raise ValueError(f"{w}: the packet's FCS ({t['fcs']}) does not check out")
            return body
        if "header" in t:
            return data[HEADERS[t["header"]][1](data):]
        if "restore" in t:
            return RESTORES[t["restore"]][1](data)
        raise ConfigError(f"{w}: transform {t}")

def _info(p):
    ok = {True: "ok", False: "FAILED", None: ""}[p.crc_ok]
    f = " ".join(f"{k}={_text(v)}" for k, v in p.fields.items())
    return " ".join(x for x in (ok + (f" ({p.check})" if p.check and ok else ""), f) if x)

# ---- a profile's rules

class Captures:
    """A profile's capture rules, by message kind (`"capture"` in a profile; see module). `packet(message)`: its Packet,
    None for a kind not captured or a message dropped. A kind without a rule: exported PDU data"""
    def __init__(self, spec=None, where="capture", params=None):
        spec = dict(spec or {})
        self.where, self.params = where, dict(params or {})
        self.rules = {k: (False if v is False else Capture(v, f"{where} {k!r}", self.params)) for k, v in spec.items()}
        self._default = None

    def rule(self, kind):
        """The rule for a kind (False: not captured)"""
        r = self.rules.get(kind)
        if r is None:
            if self._default is None:
                self._default = Capture({"linktype": "upper_pdu"}, f"{self.where} (default)", self.params)
            return self._default
        return r

    def packet(self, msg):
        r = self.rule(msg.kind)
        return None if r is False else r.packet(msg)

    def packets(self, messages):
        return [p for p in (self.packet(m) for m in messages) if p is not None]

    def linktypes(self):
        """The link types the rules give, by kind (None: found per frame)"""
        return {k: (None if r is False else r.linktype) for k, r in self.rules.items()}

    def __bool__(self):
        return bool(self.rules)
