"""Composite checks: a check field built from CRCs over transformed views of the message

    {"base": "crc", "type": "composite",
     "parts": [{"crc": "crc8_smbus"},                                   CRC-8 of the message
               {"crc": "crc8_smbus", "input": ["nibble_swap"]}],        CRC-8 of the message with each byte's nibbles swapped
     "combine": "concat",                                               the field: the parts' fields in order (or "xor")
     "check": "0x...."}                                                 optional: compute(b"123456789"), verified on load

A part is a crc definition (`crc`, a config/crc name) with optional `input` transforms, applied in order to the message
bytes before its register runs, and `output` transforms on its value. A transform is a name or {"op": name, ...args}:

    nibble_swap                 each byte's two nibbles swapped
    reverse_bits                each byte's bits reversed
    reverse_bytes               the bytes in reverse order
    invert                      every bit inverted
    {"op": "xor", "value": ...} XORed with a byte, or a list of bytes repeated over the message
    {"op": "skip", "n": 2}      the message without its first n bytes
    {"op": "take", "n": 4}      the first n bytes (a negative n: all but the last -n)
outputs: reflect (the value's bits reversed over the part's width), invert (all ones XORed on).

`register(name)` adds a transform from Python, `fn(data: np.ndarray[uint8], **args) -> np.ndarray[uint8]`, which a definition
then names. A composite has the `crc` interface (width, compute, compute_bits, field, syndrome, encode, decode), so it goes
wherever a CRC does: an HDLC frame's `fcs`, a link header's `crc` or `fcs`. Each part's field is sent the way its own CRC
sends it; for `xor` the first part's conventions apply and every part has its width.
"""
import numpy as np

from ..code.types._code import _code
from ..errors import ConfigError
from ..math.bits import reflect
from .crc import _REV8

TRANSFORMS = {}

def register(name):
    """Register a byte transform a composite check's parts can name in "input": fn(data uint8 array, **args) -> uint8 array"""
    def deco(fn):
        TRANSFORMS[name] = fn
        return fn
    return deco

@register("nibble_swap")
def _nibble_swap(d):
    return ((d << 4) | (d >> 4)).astype(np.uint8)

@register("reverse_bits")
def _reverse_bits(d):
    return _REV8[d]

@register("reverse_bytes")
def _reverse_bytes(d):
    return d[::-1]

@register("invert")
def _invert(d):
    return ~d

@register("xor")
def _xor(d, value):
    v = np.atleast_1d(np.asarray([int(x, 0) if isinstance(x, str) else int(x) for x in np.atleast_1d(value)], dtype=np.uint8))
    return d ^ np.resize(v, len(d)) if len(d) else d

@register("skip")
def _skip(d, n):
    return d[int(n):]

@register("take")
def _take(d, n):
    return d[:int(n)]

OUTPUTS = ("reflect", "invert")

def _op(spec, name):
    """(transform name, args) of an "input" entry"""
    if isinstance(spec, str):
        op, args = spec, {}
    elif isinstance(spec, dict) and "op" in spec:
        op, args = spec["op"], {k: v for k, v in spec.items() if k not in ("op", "comment")}
    else:
        raise ConfigError(f"{name}: a transform is a name or {{\"op\": name, ...}}, not {spec!r}")
    if op not in TRANSFORMS:
        raise ConfigError(f"{name}: no transform {op!r} (have {sorted(TRANSFORMS)}; rp.link.check.register adds one)")
    return op, args


class Part:
    """One CRC over a transformed view of the message"""
    def __init__(self, crc, inputs=(), outputs=(), name="composite"):
        self.crc = crc
        self.inputs = [_op(s, name) for s in inputs]
        for o in outputs:
            if o not in OUTPUTS:
                raise ConfigError(f"{name}: output {o!r} is one of {OUTPUTS}")
        self.outputs = list(outputs)
        self.width = crc.width

    def _out(self, v):
        for o in self.outputs:
            v = reflect(v, self.width) if o == "reflect" else v ^ ((1 << self.width) - 1)
        return v

    def _view(self, data):
        for op, args in self.inputs:
            data = np.asarray(TRANSFORMS[op](data, **args), dtype=np.uint8)
        return data

    def compute(self, data):
        return self._out(self.crc.compute(self._view(np.frombuffer(bytes(data), np.uint8)).tobytes()))

    def compute_bits(self, bits):
        if not self.inputs:
            return self._out(self.crc.compute_bits(bits))
        bits = np.asarray(bits, dtype=np.uint8).reshape(-1)
        if len(bits) % 8:
            raise ValueError(f"{len(bits)} bits: a part with byte transforms needs whole bytes")
        # The bytes as the CRC reads them: compute(data) == compute_bits(unpackbits(data, little if refin else big))
        data = np.packbits(bits, bitorder="little" if self.crc.refin else "big")
        return self.compute(data.tobytes())


class composite(_code):
    """A check from several CRC parts, concatenated or XORed (see the module docstring)"""
    def __init__(self, log, parts, combine="concat", check=None, name="composite"):
        super().__init__(log, None, 1, 1)
        if not parts:
            raise ConfigError(f"{name}: a composite check needs parts")
        if combine not in ("concat", "xor"):
            raise ConfigError(f"{name}: combine is concat or xor, not {combine!r}")
        if combine == "xor" and len({p.width for p in parts}) > 1:
            raise ConfigError(f"{name}: xor combines parts of one width, have {[p.width for p in parts]}")
        self.parts, self.combine = parts, combine
        self.width = sum(p.width for p in parts) if combine == "concat" else parts[0].width
        self.mask = (1 << self.width) - 1
        self.check = check
        if check is not None and self.compute(b"123456789") != check:
            raise ValueError(f"{name}: composite check fails its check value: {self.compute(b'123456789'):#x} != {check:#x}")

    def _join(self, values):
        if self.combine == "xor":
            out = 0
            for v in values:
                out ^= v
            return out
        out = 0
        for p, v in zip(self.parts, values):
            out = (out << p.width) | v
        return out

    def _split(self, value):
        """The parts' values of a concatenated value (first part in the high bits)"""
        if self.combine == "xor":
            return [value]
        out, sh = [], self.width
        for p in self.parts:
            sh -= p.width
            out.append((value >> sh) & ((1 << p.width) - 1))
        return out

    def _senders(self):
        return [p.crc for p in self.parts] if self.combine == "concat" else [self.parts[0].crc]

    def compute(self, data):
        """The check of bytes: each part's CRC of its view, combined"""
        return self._join([p.compute(data) for p in self.parts])

    def compute_bits(self, bits):
        """The check of a bit stream, in the order sent"""
        return self._join([p.compute_bits(bits) for p in self.parts])

    def to_bytes(self, value):
        return b"".join(c.to_bytes(v) for c, v in zip(self._senders(), self._split(value)))

    def to_bits(self, value):
        return np.concatenate([c.to_bits(v) for c, v in zip(self._senders(), self._split(value))])

    def from_bits(self, bits):
        bits = np.asarray(bits, dtype=np.uint8)
        values, i = [], 0
        for c in self._senders():
            values.append(c.from_bits(bits[i:i + c.width]))
            i += c.width
        return self._join(values) if self.combine == "concat" else values[0]

    def _cut(self, bits):
        bits = np.asarray(bits).astype(np.uint8).reshape(-1)
        if len(bits) < self.width:
            raise ValueError(f"{len(bits)} bits: shorter than the {self.width}-bit check")
        return bits[:-self.width], bits[-self.width:]

    def field(self, bits, mask=0):
        """The check field of message bits, as sent after them"""
        return self.to_bits(self.compute_bits(bits) ^ mask)

    def encode(self, bits, mask=0):
        bits = np.asarray(bits).astype(bool).reshape(-1)
        return np.concatenate((bits, self.field(bits, mask)))

    def syndrome(self, bits, mask=0):
        msg, field = self._cut(bits)
        return self.from_bits(field) ^ self.compute_bits(msg) ^ mask

    def _decode(self, bits, mask=0):
        msg, _ = self._cut(bits)
        return msg.astype(bool), np.array([0 if self.syndrome(bits, mask) == 0 else -1])
