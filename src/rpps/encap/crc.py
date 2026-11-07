import numpy as np

from ..coding.types._code import _code
from ..math.bits import reflect

class crc(_code):
    """Parametrised CRC (Rocksoft model: width, poly, init, refin, refout, xorout), width >= 8

    compute(data) takes bytes / a uint8 array. encode(bits) appends the CRC to byte-aligned data, decode(bits)
    strips it and sets `ok` per call (False when it doesn't check out). The CRC is appended most significant
    byte first, or least significant first when refout is set, as the common definitions do.
    """
    def __init__(self, log, width=16, poly=0x1021, init=0, refin=False, refout=False, xorout=0, check=None):
        super().__init__(log, None, 1, 1)
        self.width, self.poly, self.init = width, poly, init
        self.refin, self.refout, self.xorout = refin, refout, xorout
        self.mask = (1 << width) - 1
        self.check = check
        self.ok = None
        self.table = self._table()
        if check is not None and self.compute(b"123456789") != check:
            raise ValueError(f"CRC definition fails its check value: {self.compute(b'123456789'):#x} != {check:#x}")

    def _table(self):
        t = []
        if self.refin:
            rp = reflect(self.poly, self.width)
            for i in range(256):
                c = i
                for _ in range(8):
                    c = (c >> 1) ^ (rp if c & 1 else 0)
                t.append(c)
        else:
            top = 1 << (self.width - 1)
            for i in range(256):
                c = i << (self.width - 8)
                for _ in range(8):
                    c = ((c << 1) ^ self.poly if c & top else c << 1) & self.mask
                t.append(c)
        return t

    def compute(self, data):
        c = reflect(self.init, self.width) if self.refin else self.init
        for b in bytes(data):
            if self.refin:
                c = (c >> 8) ^ self.table[(c ^ b) & 0xFF]
            else:
                c = ((c << 8) & self.mask) ^ self.table[((c >> (self.width - 8)) ^ b) & 0xFF]
        if self.refin != self.refout:
            c = reflect(c, self.width)
        return (c ^ self.xorout) & self.mask

    def to_bytes(self, value):
        n = self.width // 8
        return value.to_bytes(n, "little" if self.refout else "big")

    def encode(self, bits):
        data = np.packbits(np.asarray(bits, dtype=np.uint8))
        return np.unpackbits(np.concatenate((data, np.frombuffer(self.to_bytes(self.compute(data)), np.uint8)))).astype(bool)

    def decode(self, bits):
        by = np.packbits(np.asarray(bits, dtype=np.uint8))
        n = self.width // 8
        data, tail = by[:-n], bytes(by[-n:])
        self.ok = self.to_bytes(self.compute(data)) == tail
        return np.unpackbits(data).astype(bool)
