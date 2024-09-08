from .__bitarray import bitarray

class Stream:
    __slots__ = ("cache", "length")

    def __init__(self):
        self.cache = {
            "bin": None,
            "hex": None,
            "bytes": None,
            "int": None,
            "bitarray": None,
        }
        self.length = None

    def __str__(self):
        if self.cache["hex"] is not None:
            return f"0x{self.hex}"
        return str({k: str(v) for k, v in self.cache.items()})

    def __len__(self):
        return self.length if self.length is not None else 0

    def __repr__(self):
        return f"<Stream data>"

    @property
    def bin(self):
        if self.cache["bin"] is None:
            # self.cache["bin"] = Encoding._pad(bin(self.cache["int"])[2:], 8)
            self.cache["bin"] = "".join([item for item in self.bitarray.astype(str)])
        return self.cache["bin"]

    @bin.setter
    def bin(self, data):
        self.cache["bin"] = data

    @property
    def hex(self):
        if self.cache["hex"] is None:
            self.cache["hex"] = Stream._pad(hex(self.cache["int"])[2:], 2)
        return self.cache["hex"]

    @hex.setter
    def hex(self, data):
        self.cache["hex"] = data

    @property
    def bytes(self):
        if self.cache["bytes"] is None:
            if self.cache["hex"] is None:
                return None
            self.cache["bytes"] = bytes.fromhex(self.cache["hex"])
        return self.cache["bytes"]

    @bytes.setter
    def bytes(self, data):
        self.cache["bytes"] = data

    @property
    def int(self):
        return self.cache["int"]

    @property
    def bitarray(self):
        if self.cache["bitarray"] is None:
            self.cache["bitarray"] = bitarray.from_encoding(self)
        return self.cache["bitarray"]

    @staticmethod
    def from_bin(binary, length = None):
        enc = Stream()
        enc.cache["bin"] = Stream._pad(binary, 8)
        enc.cache["int"] = int(binary, 2)
        enc.length = length
        return enc

    @staticmethod
    def from_hex(hexadecimal, length = None):
        enc = Stream()
        enc.cache["hex"] = Stream._pad(hexadecimal, 2)
        enc.cache["int"] = int(hexadecimal, 16)
        enc.length = length
        return enc

    @staticmethod
    def from_bytes(byte, length = None):
        enc = Stream()
        enc.cache["bytes"] = byte
        enc.cache["int"] = int.from_bytes(byte)
        enc.length = length
        return enc

    @staticmethod
    def from_int(int, length = None):
        enc = Stream()
        enc.cache["int"] = int
        enc.length = length
        return enc

    @staticmethod
    def from_bitarray(ba, length=None):
        enc = Stream()
        enc.cache["bitarray"] = ba
        enc.length = length
        return enc

    @staticmethod
    def _pad(data, mod, fill = "0"):
        padding = len(data) % mod
        if not padding == 0:
            data = fill * (mod - padding) + data
        return data

    def slice(self, field, size):
        idx = 0
        while idx < len(self.cache[field]):
            yield self.cache[field][idx:idx+size]
            idx += size
