import numpy as np

class bitarray:
    def __init__(self, arr=None):
        if arr is None:
            arr = np.array([], dtype=np.bool)
        self.arr = arr

    def __str__(self):
        return str(self.arr)

    def __len__(self):
        return len(self.arr)

    def append(self, bit):
        self.arr = np.append(self.arr, bit)

    @staticmethod
    def from_encoding(enc):
        return bitarray(
            np.unpackbits(
                np.array(bytearray(enc.bytes), dtype=np.uint8)
            )
        )

    def to_bytes(self) -> bytes:
        return bytes(np.packbits(self.arr))

    def bits(self):
        return self.arr

    def bytes(self):
        return np.packbits(self.arr)

    def astype(self, *args, **kwargs):
        return self.arr.astype(*args, **kwargs)

    def __iter__(self):
        return self.arr.__iter__()

    def __getattr__(self, key):
        return self.arr

    def __getitem__(self, key):
        return self.arr[key]

    def __setitem__(self, key, val):
        self.arr[key] = val
