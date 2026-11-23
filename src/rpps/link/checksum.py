"""Checksums that are not CRCs: sums, ones' complement (Internet), Fletcher, Adler, XOR (LRC)

    {"base": "crc", "type": "checksum", "algorithm": "internet",   sum | internet | fletcher | adler | xor
     "width": 16,                                                   bits of the value (sum, xor: 8, 16 or 32)
     "word": 16, "endian": "big",                                   sum, xor: the message read as words of this many bits
     "negate": false,                                               sum: the two's complement of the sum (Intel HEX, NMEA's
                                                                    is xor) so that message + checksum sums to 0
     "xorout": "0x0",                                               XORed onto the value
     "refin": false,                                                the bytes are sent least significant bit first
     "order": "big",                                                the value's bytes as sent after the message
     "check": "0x...."}                                             optional: compute(b"123456789"), verified on load

    internet  RFC 1071: the ones' complement of the ones' complement sum of big-endian 16-bit words (a last odd byte
              padded with a zero)
    fletcher  width 16: two sums of bytes mod 255 (value sum2 << 8 | sum1); width 32: of 16-bit words (`endian`) mod 65535
    adler     Adler-32 (zlib): sums of bytes mod 65521, value s2 << 16 | s1

It has the `crc` interface (width, compute, compute_bits, to_bytes, to_bits, field, syndrome, encode, decode), so it goes
wherever a CRC does: an HDLC or byte-stuffed frame's `fcs`, a link header's `crc` or `fcs`, a composite check's part.
"""
import numpy as np

from ..code.types._code import _code
from ..errors import ConfigError

ALGORITHMS = ("sum", "internet", "fletcher", "adler", "xor")

class checksum(_code):
    def __init__(self, log, algorithm="sum", width=8, word=8, endian="big", negate=False, xorout=0, refin=False,
                 order="big", check=None, name="checksum"):
        super().__init__(log, None, 1, 1)
        if algorithm not in ALGORITHMS:
            raise ConfigError(f"{name}: algorithm is one of {ALGORITHMS}, not {algorithm!r}")
        if algorithm == "internet":
            width, word = 16, 16
        elif algorithm == "adler":
            width = 32
        elif algorithm == "fletcher" and width not in (16, 32):
            raise ConfigError(f"{name}: Fletcher is 16 or 32 bits wide, not {width}")
        if algorithm in ("sum", "xor") and (width not in (8, 16, 32) or word not in (8, 16, 32)):
            raise ConfigError(f"{name}: width and word are 8, 16 or 32 bits")
        if endian not in ("big", "little") or order not in ("big", "little"):
            raise ConfigError(f"{name}: endian and order are big or little")
        self.algorithm, self.width, self.word, self.endian, self.negate = algorithm, width, word, endian, negate
        self.xorout, self.refin, self.order = xorout, refin, order
        self.mask = (1 << width) - 1
        self.check = check
        if check is not None and self.compute(b"123456789") != check:
            raise ValueError(f"{name}: checksum fails its check value: {self.compute(b'123456789'):#x} != {check:#x}")

    def _words(self, data, bits):
        n = bits//8
        data = bytes(data) + bytes(-len(data) % n) # a last part word padded with zeros
        return [int.from_bytes(data[i:i + n], self.endian) for i in range(0, len(data), n)]

    def compute(self, data):
        """The checksum of bytes"""
        data = bytes(data)
        a = self.algorithm
        if a == "sum":
            v = sum(self._words(data, self.word)) & self.mask
            v = (-v) & self.mask if self.negate else v
        elif a == "xor":
            v = 0
            for w in self._words(data, self.word):
                v ^= w
            v &= self.mask
        elif a == "internet":
            s = sum(int.from_bytes(data[i:i + 2].ljust(2, b"\0"), "big") for i in range(0, len(data), 2))
            while s >> 16:
                s = (s & 0xFFFF) + (s >> 16)
            v = ~s & 0xFFFF
        elif a == "fletcher":
            half, mod = (8, 255) if self.width == 16 else (16, 65535)
            s1 = s2 = 0
            for w in (data if half == 8 else self._words(data, 16)):
                s1 = (s1 + w) % mod
                s2 = (s2 + s1) % mod
            v = s2 << half | s1
        else: # adler
            s1, s2 = 1, 0
            for b in data:
                s1 = (s1 + b) % 65521
                s2 = (s2 + s1) % 65521
            v = s2 << 16 | s1
        return (v ^ self.xorout) & self.mask

    def compute_bits(self, bits):
        """The checksum of a bit stream in the order sent: whole bytes, each least significant bit first with `refin`"""
        bits = np.asarray(bits, dtype=np.uint8).reshape(-1)
        if len(bits) % 8:
            raise ValueError(f"{len(bits)} bits: a checksum is over whole bytes")
        return self.compute(np.packbits(bits, bitorder="little" if self.refin else "big").tobytes())

    def to_bytes(self, value):
        return int(value).to_bytes(self.width//8, self.order)

    def to_bits(self, value):
        return np.unpackbits(np.frombuffer(self.to_bytes(value), np.uint8), bitorder="little" if self.refin else "big").astype(bool)

    def from_bits(self, bits):
        b = np.packbits(np.asarray(bits, dtype=np.uint8), bitorder="little" if self.refin else "big").tobytes()
        return int.from_bytes(b, self.order)

    def _cut(self, bits):
        bits = np.asarray(bits).astype(np.uint8).reshape(-1)
        if len(bits) < self.width:
            raise ValueError(f"{len(bits)} bits: shorter than the {self.width}-bit checksum")
        return bits[:-self.width], bits[-self.width:]

    def field(self, bits, mask=0):
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
