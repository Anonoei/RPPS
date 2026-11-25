"""Encapsulation parent classes"""
import numpy as np

from ..logger import Logger
from ..errors import ConfigError

from .. import _config
from ..code.coding import Coding
from ..base.rpps import FLUSH
from ..base.types import Category, Rep
from .crc import crc as _crc_code
from .hdlc import hdlc as _hdlc_code


class Encap(Coding):
    """Parent of framing / integrity-check definitions: the link transform, data bits <-> packets"""
    kind = "encap"
    CATEGORY = Category.LINK
    ENCODE_TAKES, DECODE_TAKES = {Rep.DATA, Rep.PACKETS}, {Rep.DATA} # Decoding refuses channel bits: decode them first
    ENCODE_GIVES, DECODE_GIVES = Rep.DATA, Rep.PACKETS
    CONTROLS = {}

    def pos_rate(self, decode):
        return None # Packets of varying length: measured

    @staticmethod
    def load(name: str, obj: dict):
        """Load from json: the `type` picks CRC or HDLC"""
        t = obj.get("type")
        if t in ("crc", "composite", "checksum"):
            return CRC.load(name, obj)
        if t in ("hdlc", "bytes"):
            return Frame.load(name, obj)
        raise ConfigError(f"Unknown encapsulation type {t!r} in {name}: have crc, composite, checksum, hdlc, bytes")


def _int(obj, key, default=0):
    v = obj.get(key, default)
    return int(v, 0) if isinstance(v, str) else v


def _hex(v):
    return int(v, 0) if isinstance(v, str) else int(v)

def _flat(v):
    """A list with the lists in it (an "each" range among single values) spread out"""
    return [y for x in v for y in (_flat(x) if isinstance(x, list) else [x])]


class CRC(Encap):
    """CRC defined by config: width, poly, init, refin, refout, xorout (hex strings or ints), check value; or a composite
    check (type "composite": CRC parts over transformed views of the message, rpps.link.check)"""

    @staticmethod
    def load(name, obj):
        log = Logger().Child("Coding", _config.LOG_CODING).Child(name)
        if obj.get("type") == "composite":
            i_code = _composite(log, name, obj)
        elif obj.get("type") == "checksum":
            from .checksum import checksum
            i_code = checksum(log, obj.get("algorithm", "sum"), obj.get("width", 8), obj.get("word", 8), obj.get("endian", "big"),
                              obj.get("negate", False), _int(obj, "xorout"), obj.get("refin", False), obj.get("order", "big"),
                              _int(obj, "check") if "check" in obj else None, f"crc {name}")
        else:
            i_code = _crc_code(log, obj["width"], _int(obj, "poly"), _int(obj, "init"), obj.get("refin", False),
                               obj.get("refout", False), _int(obj, "xorout"), _int(obj, "check") if "check" in obj else None)
        impl = type(name, (CRC,), dict())
        impl.name = name
        return impl(i_code, i_code)


def _composite(log, name, obj):
    from . import serial
    from .check import composite, Part
    parts = []
    for i, p in enumerate(obj.get("parts") or []):
        if not isinstance(p, dict) or "crc" not in p:
            raise ConfigError(f"crc {name}: part {i} needs a \"crc\" (a config/crc name)")
        parts.append(Part(serial.load(f"crc/{p['crc']}").encoder, p.get("input", ()), p.get("output", ()), f"crc {name}"))
    return composite(log, parts, obj.get("combine", "concat"), _int(obj, "check") if "check" in obj else None, f"crc {name}")


class Frame(Encap):
    """Framing (HDLC: bit stuffing; bytes: byte stuffing, SLIP / KISS / PPP) defined by config; the FCS is another config,
    `fcs`, from config/crc

    As an operator: `packet * frame` gives one frame's bits (flags, stuffing, FCS) per call; `bits / frame` the payload bits of
    the frames whose FCS checks out, however the stream is cut (an open frame waits for its closing flag in the next block)

    Receiving, each good frame's payload starts with a publication, {"frame": where the frame starts, "frame_bits": its
    payload's bits}, at its place in the output, and an anchor there: a step after it (a link header that strips the
    frame's own header, `{"link": "header"}` driven by `"frame"` and `"length": "frame_bits"`) reads frame by frame.
    Every frame's message carries the whole frame as received, payload then FCS (`Message.frame`, `parts`)"""
    EMITS = True # Every frame, good or not, is a message (rpps.profile.message)
    payload_schema = None # "payload_fields": what a good frame's payload says, in its message (rpps.link.fields)
    check_name = None # The FCS's definition (Message.check)

    def _block(self, enc):
        return 8 if enc else 1

    def _process(self, coder, x, ctx, enc):
        if enc:
            if not len(x):
                return np.zeros(0, dtype=bool), None
            return coder.encode([np.packbits(np.asarray(x, dtype=np.uint8)).tobytes()]), None
        key = (self, "held")
        prev = ctx.pop(key, np.zeros(0, np.uint8))
        frames = []
        out, held = coder.decode_stream(x, prev, frames)
        base = len(prev) # frames_at indexes held + x: x[0] is at the block's position
        if ctx.get(FLUSH):
            rest = coder.decode(held) if len(held) else np.zeros(0, bool)
            out = np.concatenate((out, rest))
            if len(held):
                skip = len(prev) + len(x) - len(held)
                frames += [(s + skip, p, ok, fcs) for s, p, ok, fcs in coder.frames_at(held)[0]]
        else:
            ctx[key] = held
        ctx[(self, "frames")] = [(s - base, p, ok, fcs) for s, p, ok, fcs in frames] # For decode, which emits them
        return out, None # (only the frames whose FCS checks out come out)

    def decode(self, data):
        """`bits / frame`: the good frames' payloads; every frame, good or not, also emitted as a message from its opening
        flag (rpps.profile.message: bytes, crc_ok the FCS's verdict)"""
        out = super().decode(data)
        from ..base.control import emit, publish, anchor
        n_out = 0 # Where each good frame's payload starts in the output (they come out in order)
        for s, p, ok, fcs in out.ctx.pop((self, "frames"), []):
            bits = np.unpackbits(np.frombuffer(p, np.uint8)).astype(bool)
            at = data.meta.pos + s
            fields = {"bytes": len(p)}
            if ok and self.payload_schema is not None: # What the payload says ("payload_fields": rpps.link.fields)
                from .fields import Short
                try:
                    fields["payload"] = self.payload_schema.parse(bits)[0]
                except (Short, ValueError, ConfigError):
                    pass
            whole = np.unpackbits(np.frombuffer(p + fcs, np.uint8)).astype(bool)
            parts = {"payload": (0, len(bits))}
            if fcs:
                parts["fcs"] = (len(bits), 8*len(fcs))
            emit(data.meta, self, None, fields, bits, bool(ok), at, frame=whole, parts=parts)
            if ok:
                publish(data.meta, self, {"frame": at, "frame_bits": len(bits)}, at=at, out=n_out, group=at)
                anchor(data.meta, self, at, n_out)
                n_out += len(bits)
        return out

    @staticmethod
    def load(name, obj):
        from . import serial
        log = Logger().Child("Coding", _config.LOG_CODING).Child(name)
        if obj.get("type") == "bytes":
            from .bytestuff import bytestuff
            fcs = serial.load(f"crc/{obj['fcs']}").encoder if obj.get("fcs") else None
            esc = {_hex(k): _hex(v) for k, v in obj["escapes"].items()} if obj.get("escapes") else None
            i_code = bytestuff(log, _int(obj, "flag", 0xC0), _int(obj, "escape", 0xDB), esc,
                               _int(obj, "xor") if "xor" in obj else None,
                               [_hex(b) for b in _flat(obj["escaped"])] if "escaped" in obj else None, fcs, obj.get("refin", False),
                               obj.get("min_bytes", 1))
        else:
            fcs = serial.load(f"crc/{obj['fcs']}").encoder
            i_code = _hdlc_code(log, fcs, _int(obj, "flag", 0x7E), obj.get("stuff", 5), obj.get("nrzi", False),
                                obj.get("min_bytes", 4))
        impl = type(name, (Frame,), dict())
        impl.name = name
        impl.check_name = obj.get("fcs")
        if obj.get("payload_fields"):
            from .fields import Schema
            impl.payload_schema = Schema(obj["payload_fields"], "msb", f"frame {name} payload")
        return impl(i_code, i_code)
