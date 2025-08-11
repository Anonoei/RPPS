"""Data Containers

DSP (Analog)
  Samples [bytes] - Raw IQ samples
    Fs, sps (sr, cf)
  Symbols [bytes] - Synchronized IQ symbols
(De)Modulation (Conversion)
  Mapped [bits/bytes] - Mapped symbols
    Hard/Soft
Coding (Digital)
  Coded [bits/bytes] - Coding data (ECC)
  Scram [bits/bytes] - Scrambler data
Communication
  Framing [bits/bytes] - Framing data
"""

from enum import Enum, auto
import numpy as np
from . import types
from .types import dtype, ptype


def Data(data):
    d = None
    if isinstance(data, np.ndarray):
        if data.dtype == bool:
            d = Digital(data, dt=dtype.bits)
        elif data.dtype == np.uint8:
            d = Digital(data, dt=dtype.bytes)
        elif data.dtype == dtype.samples.value:
            d = Analog(data, pt=ptype.SAMPLES, dt=dtype.samples)
        elif data.dtype == dtype.symbols.value:
            d = Analog(data, pt=ptype.SYMBOLS, dt=dtype.samples)
        else:
            raise NotImplementedError(f"Cannot convert {type(data)} {data.shape} {data.dtype}")
    elif isinstance(data, bytes):
        d = Digital.from_bytes(data, ptype.MSG, dtype.bytes)
    elif isinstance(data, str):
        d = Digital.from_bytes(data.encode("utf-8"), ptype.MSG, dtype.bytes)
    if d is None:
        raise NotImplementedError(f"Cannot convert {type(data)}")
    print(f"Converted {type(data)} to {d}")
    return d

class _Data:
    __slots__ = (
        "_data",
        "PT", "DT"
    )

    def __init__(self, data=None, pt = ptype.UNKNOWN, dt=None):
        self._data: np.ndarray = data # type: ignore
        self.PT = pt
        if dt is not None:
            self.DT = dt
        elif data is not None:
            self.DT = data.dtype
        else:
            self.DT = None

    @property
    def name(self):
        """Get class name"""
        return type(self).__name__

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data):
        self._data = data

    def __str__(self):
        return f"({self.PT}/{self.DT}) {self.name}"

    def __len__(self):
        return len(self._data)

    def __enter__(self, *args, **kwargs):
        return self._data.__enter__(*args, **kwargs) # type: ignore

    def __exit__(self, *args, **kwargs):
        return self._data.__exit__(*args, **kwargs) # type: ignore

    def __iter__(self, *args, **kwargs):
        return self._data.__iter__(*args, **kwargs)

    def __getitem__(self, *args, **kwargs):
        return self._data.__getitem__(*args, **kwargs)

    def append(self, data):
        self._data = np.append(self._data, data)

class Analog(_Data):
    pass

class Mod(_Data):
    __slots__ = (
        "code", "dist",
        "decided"
    )
    def __init__(self, data=None):
        if isinstance(data, tuple):
            codewords = data[0]
            distances = data[1]
            self._data = None
            self.PT = ptype.MAPPED_SOFT
            self.DT = dtype.mapped_soft
            self.decided = False
        else:
            codewords = None
            distances = None
            self._data: np.ndarray = data # type: ignore
            self.PT = ptype.MAPPED_HARD
            self.DT = dtype.mapped_hard
            self.decided = True

        self.code = codewords
        self.dist = distances

    def decide(self, code_idx):
        if self.PT == ptype.MAPPED_SOFT:
            if self._data is None:
                self._data = np.zeros((1, self.code.shape[1]), dtype=dtype.bits.value) # type: ignore
                self._data[0] = self.code[code_idx] # type: ignore
                return self._data[-1]
        elif self._data.shape[0] < self.dist.shape[0]: # type: ignore
            self._data = np.append(self._data, self.code[code_idx], axis=1) # type: ignore
            if self._data.shape[0] == self.dist.shape[0]: # type: ignore
                self.decided = True
            return self._data[-1]
        return None

    def as_hard(self):
        if self.PT == ptype.MAPPED_SOFT:
            max_vals = np.max(self.dist, axis=1) # type: ignore
            indices = np.argwhere(np.equal(self.dist, max_vals[:, None])) # type: ignore
            self._data = self.code[indices[:,1]].reshape(-1).astype(bool) # type: ignore
            self.code = None
            self.dist = None
            self.PT = ptype.MAPPED_HARD
            self.DT = dtype.mapped_hard
        return self

class Digital(_Data):
    @property
    def bin(self):
        if self.is_bit():
            return self._data.astype(int)
        raise ValueError(f"({self.PT}/{self.DT}) {self._data.dtype} cannot be shown in binary")

    @property
    def hex(self):
        if self.is_byte():
            return self._data.tobytes().hex()
        raise ValueError(f"({self.PT}/{self.DT}) {self._data.dtype} cannot be shown in hex")

    def is_bit(self):
        if isinstance(self.DT, dtype):
            return self.DT == dtype.bits
        else:
            return self.DT == dtype.bits.value
    def is_byte(self):
        if isinstance(self.DT, dtype):
            return self.DT == dtype.bytes
        else:
            return self.DT == dtype.bytes.value

    def as_bytes(self):
        if not self.is_byte():
            self._data = self.get_bytes() # type: ignore
            self.DT = dtype.bytes
        return self

    def as_bits(self):
        if not self.is_bit():
            self._data = self.get_bits() # type: ignore
            self.DT = dtype.bits
        return self

    def get_bytes(self):
        if self.is_bit():
            return np.packbits(self._data)
        elif self.is_byte():
            return self._data.astype(dtype.bytes.value)
        else:
            raise ValueError(f"({self.PT}/{self.DT}) {self._data.dtype} cannot get bytes")

    def get_bits(self):
        if self.is_bit():
            return self._data.astype(dtype.bits.value)
        elif self.is_byte():
            return np.unpackbits(self._data).astype(dtype.bits.value)
        else:
            raise ValueError(f"({self.PT}/{self.DT}) {self._data.dtype} cannot get bits")

    @classmethod
    def from_bytes(cls, data, pt=ptype.UNKNOWN, dt=dtype.bytes):
        if isinstance(data, bytes):
            return cls(np.frombuffer(data, dtype=dtype.bytes.value), pt, dt)
        elif isinstance(data, np.ndarray):
            if data.dtype == np.uint8:
                return cls(data, pt, dt)
            raise NotImplementedError(f"Cannot convert {type(data)} {data.shape} {data.dtype} to {cls.__name__}")
        raise NotImplementedError(f"Cannot convert {type(data)} to {cls.__name__}")

    @classmethod
    def from_bits(cls, data, pt=ptype.UNKNOWN, dt=dtype.bits):
        if isinstance(data, np.ndarray):
            if data.dtype == np.bool:
                return cls(data, pt, dt)
            raise NotImplementedError(f"Cannot convert {type(data)} {data.shape} {data.dtype} to {cls.__name__}")
        raise NotImplementedError(f"Cannot convert {type(data)} to {cls.__name__}")
