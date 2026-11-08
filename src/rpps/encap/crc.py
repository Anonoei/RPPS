import numpy as np

from ..coding.types._code import _code
from ..math.bits import reflect

_REV8 = np.array([reflect(i, 8) for i in range(256)], dtype=np.uint8) # Byte with its bits reversed

class crc(_code):
    """Parametrised CRC (Rocksoft model: width, poly, init, refin, refout, xorout), any width from 1 to 64

    Two views of the same register:
      compute(data)       bytes / a uint8 array, as the catalogues define it: with `refin` each byte goes in least
                          significant bit first. compute(b"123456789") is the catalogue's check value.
      compute_bits(bits)  a bit stream of any length, fed in the order given (the order it is sent). `refin` is then
                          already in the order of the bits; `refout` and `xorout` apply as usual, so
                          compute(data) == compute_bits(np.unpackbits(data, bitorder="little" if refin else "big")).

    encode(bits) appends the CRC to a message of any length and decode(bits) strips it, setting `ok`. Both work on the
    bit stream as sent: the CRC follows the message bits the way the register would shift it out, most significant
    bit first, or least significant first when `refout` is set (HDLC's FCS, BLE). A received frame passes when the CRC
    over the message equals the CRC field.

    mask: XORed onto the CRC value by encode / decode / syndrome, for standards that overlay something on it (LTE's
    antenna port mask or RNTI, the Mode S address). syndrome(bits) is the received CRC field XOR the computed one: 0 for
    a good frame, the overlay (or an error pattern) otherwise.
    """
    def __init__(self, log, width=16, poly=0x1021, init=0, refin=False, refout=False, xorout=0, check=None):
        super().__init__(log, None, 1, 1)
        if not 1 <= width <= 64:
            raise ValueError(f"CRC width {width}: need 1 to 64 bits")
        self.width, self.poly, self.init = width, poly, init
        self.refin, self.refout, self.xorout = refin, refout, xorout
        self.mask = (1 << width) - 1
        self.check = check
        self.ok = None
        self.table = self._table() if width >= 8 else None
        if check is not None and self.compute(b"123456789") != check:
            raise ValueError(f"CRC definition fails its check value: {self.compute(b'123456789'):#x} != {check:#x}")

    def _table(self):
        """Register update for one byte, most significant bit first (the register in its unreflected form)"""
        t = []
        top = 1 << (self.width - 1)
        for i in range(256):
            c = i << (self.width - 8)
            for _ in range(8):
                c = ((c << 1) ^ self.poly if c & top else c << 1) & self.mask
            t.append(c)
        return t

    # ------------------------------------------------------------------ register
    def _feed_bytes(self, c, data):
        """Shift whole bytes into the register, most significant bit first"""
        if self.table is None:
            return self._feed_bits(c, np.unpackbits(np.frombuffer(bytes(data), np.uint8)))
        sh, m, t = self.width - 8, self.mask, self.table
        for b in bytes(data):
            c = ((c << 8) & m) ^ t[((c >> sh) ^ b) & 0xFF]
        return c

    def _feed_bits(self, c, bits):
        top, m, p = self.width - 1, self.mask, self.poly
        for b in bits:
            fb = ((c >> top) ^ int(b)) & 1
            c = (c << 1) & m
            if fb:
                c ^= p
        return c

    def _finish(self, c):
        if self.refout:
            c = reflect(c, self.width)
        return (c ^ self.xorout) & self.mask

    def compute(self, data):
        """CRC of bytes (catalogue convention: `refin` feeds each byte least significant bit first)"""
        data = np.frombuffer(bytes(data), np.uint8)
        if self.refin:
            data = _REV8[data]
        return self._finish(self._feed_bytes(self.init, data))

    def compute_bits(self, bits):
        """CRC of a bit stream of any length, bits fed in the order given"""
        bits = np.asarray(bits, dtype=np.uint8).reshape(-1)
        whole = len(bits)//8*8
        c = self._feed_bytes(self.init, np.packbits(bits[:whole]))
        return self._finish(self._feed_bits(c, bits[whole:]))

    # ------------------------------------------------------------------ the CRC field in a bit stream
    def to_bytes(self, value):
        """The CRC as bytes, the way catalogue protocols append it to byte data (little-endian when refout)"""
        n = -(-self.width//8)
        return value.to_bytes(n, "little" if self.refout else "big")

    def to_bits(self, value):
        """The CRC field as sent after the message bits"""
        order = np.arange(self.width) if self.refout else np.arange(self.width - 1, -1, -1)
        return ((int(value) >> order) & 1).astype(bool)

    def from_bits(self, bits):
        """Value of a received CRC field (inverse of to_bits)"""
        bits = np.asarray(bits, dtype=np.uint8)
        order = np.arange(self.width) if self.refout else np.arange(self.width - 1, -1, -1)
        return int(sum(int(b) << int(i) for b, i in zip(bits, order)))

    def _split(self, bits):
        bits = np.asarray(bits).astype(np.uint8).reshape(-1)
        if len(bits) < self.width:
            raise ValueError(f"{len(bits)} bits: shorter than the {self.width}-bit CRC")
        return bits[:-self.width], bits[-self.width:]

    def encode(self, bits, mask=0):
        """Message bits followed by their CRC field"""
        bits = np.asarray(bits).astype(bool).reshape(-1)
        return np.concatenate((bits, self.to_bits(self.compute_bits(bits) ^ mask)))

    def syndrome(self, bits, mask=0):
        """Received CRC field XOR the CRC of the message (and `mask`): 0 when the frame checks out"""
        msg, field = self._split(bits)
        return self.from_bits(field) ^ self.compute_bits(msg) ^ mask

    def decode(self, bits, mask=0):
        """The message without its CRC field; `ok` says whether it checked out"""
        msg, _ = self._split(bits)
        self.ok = self.syndrome(bits, mask) == 0
        self.errors = np.array([0 if self.ok else -1])
        return msg.astype(bool)
