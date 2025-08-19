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

import numpy as np
import typing

from .types import pproc, dtype
from .buffer import Buffer, Vector

class _Data:
    __slots__ = (
        "_data",
        "PP", "DT"
    )

    def __init__(self, data=None, pp=pproc.UNK, dt=None):
        self._data: np.ndarray = data # type: ignore
        # self._data = Vector(data)
        self.PP = pp
        self.DT = dt

    def __str__(self):
        return f"{self.name} ({self.PP}/{self.DT}) {len(self)}"

    def __len__(self):
        if self._data is None:
            return 0
        return len(self._data)

    def __call__(self, size):
        """
        This is a jank way to ensure the correct amount of data is piped
        Do do this correctly requires each pipe to have an input buffer and
          wait to process until the buffer is the correct size
        """
        # print(f"Size: {size}")
        rin = np.array([t[0] for t in size])
        rou = np.array([t[1] for t in size])

        mul_in = np.empty(len(size))
        mul_in[0] = rin[0]
        mul_in[1:] = rin[1:] * rou[:-1]

        mul_ou = np.empty(len(size))
        mul_ou[0] = rou[0]
        mul_ou[1:] = rou[1:] * rin[:-1]

        mul = mul_in[1:]/mul_ou[:-1]
        min_req = int(np.ceil(np.prod(mul)))
        if self.DT == dtype.BYTES: # TODO: make sure this works correctly
            min_req = int(min_req/8)
        return type(self)(self._data[:len(self)//min_req], self.PP, self.DT)

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

    def __enter__(self, *args, **kwargs):
        return self._data.__enter__(*args, **kwargs) # type: ignore

    def __exit__(self, *args, **kwargs):
        return self._data.__exit__(*args, **kwargs) # type: ignore

    def __iter__(self, *args, **kwargs):
        return self._data.__iter__(*args, **kwargs)

    def __getitem__(self, *args, **kwargs):
        return self._data.__getitem__(*args, **kwargs)

    def append(self, data):
        """Append to data"""
        self._data = np.append(self._data, data)

    def copy(self):
        """Create a copy of data"""
        return type(self)(self._data, self.PP, self.DT)
        # return type(self)(self._data._stor[:len(self._data)], self.PP, self.DT)

class Analog(_Data):
    """Analog data"""
    pass

class Mod(_Data):
    """(De)Modulation data"""
    __slots__ = (
        "code", "dist",
        "decided"
    )
    def __init__(self, data=None, pp=pproc.MAP, dt=None):
        if pp == pproc.MOD:
            self._data = data # type: ignore
            self.PP = pp
            self.DT = dt

            self.code = None
            self.dist = None
            self.decided = None
        elif pp == pproc.MAP:
            if isinstance(data, tuple):
                codewords = data[0]
                distances = data[1]
                self._data = None  # type: ignore
                self.PP = pproc.MAP
                self.DT = dtype.MAPPED_SOFT
                self.decided = False
            else:
                codewords = None
                distances = None
                self._data: np.ndarray = data # type: ignore
                self.PP = pproc.MAP
                self.DT = dtype.MAPPED_HARD
                self.decided = True

            self.code = codewords
            self.dist = distances
        else:
            raise ValueError(f"Cannot initialize Mod from {pp}")

    def get_soft(self):
        """Get codewords and distances"""
        return (self.code, self.dist)

    def to_analog(self):
        """Convert to Analog"""
        if self.PP == pproc.MOD:
            return Analog(self._data, self.PP, self.DT)
        raise ValueError(f"Cannot convert Mod {self.PP}/{self.DT} to Analog!")

    def to_digital(self):
        """Convert to Digital"""
        if self.PP == pproc.MAP:
            self.as_hard()
            return Digital(self._data, pp=pproc.MAP, dt=dtype.BITS)
        raise ValueError(f"Cannot convert Mod {self.PP}/{self.DT} to Digital!")

    def decide(self, code_idx):
        """Make hard decisions"""
        if self.DT == dtype.MAPPED_SOFT:
            if self._data is None:
                self._data = np.zeros((1, self.code.shape[1]), dtype=dtype.BITS.dtype) # type: ignore
                self._data[0] = self.code[code_idx] # type: ignore
                return self._data[-1]
            elif self._data.shape[0] < self.dist.shape[0]: # type: ignore
                self._data = np.append(self._data, self.code[code_idx], axis=1) # type: ignore
                if self._data.shape[0] == self.dist.shape[0]: # type: ignore
                    self.decided = True
                return self._data[-1]
        return None

    def as_hard(self):
        """Convert to hard map"""
        if self.DT == dtype.MAPPED_SOFT:
            max_vals = np.max(self.dist, axis=1) # type: ignore
            indices = np.argwhere(np.equal(self.dist, max_vals[:, None])) # type: ignore
            self._data = self.code[indices[:,1]].reshape(-1).astype(bool) # type: ignore
            self.code = None
            self.dist = None
            self.DT = dtype.MAPPED_HARD
        return self

    def as_bits(self):
        """Covert to Digital as bits"""
        return self.to_digital().as_bits()

    def as_bytes(self):
        """Convert to Digital as bytes"""
        return self.to_digital().as_bytes()

class Digital(_Data):
    """Digital data"""
    @property
    def bin(self):
        """Return binary representation"""
        if self.is_bit():
            return self._data.astype(int)
        raise ValueError(f"({self.PP}/{self.DT}) {self._data.dtype} cannot be shown in binary")

    @property
    def hex(self):
        """Return hexadecimal implementation"""
        if self.is_byte():
            return self._data.tobytes().hex()
        raise ValueError(f"({self.PP}/{self.DT}) {self._data.dtype} cannot be shown in hex")

    def is_bit(self, strict=True):
        """Check if data is bits"""
        idt = self.DT == dtype.BITS
        ndt = self._data.dtype == dtype.BITS.dtype
        if strict:
            return idt and ndt
        else:
            return idt or ndt
    def is_byte(self, strict=True):
        """Check if data is bytes"""
        idt = self.DT == dtype.BYTES
        ndt = self._data.dtype == dtype.BYTES.dtype
        if strict:
            return idt and ndt
        else:
            return idt or ndt

    def as_bits(self):
        """Return self as bits"""
        if not self.is_bit():
            self._data = self.get_bits() # type: ignore
            self.DT = dtype.BITS
        return self
    def as_bytes(self):
        """Return self as bytes"""
        if not self.is_byte():
            self._data = self.get_bytes() # type: ignore
            self.DT = dtype.BYTES
        return self


    def get_bytes(self):
        """Return self.data as bytes"""
        if self.is_bit(False):
            return np.packbits(self._data)
        elif self.is_byte(False):
            return self._data
        else:
            raise ValueError(f"({self.PP}/{self.DT}) {self._data.dtype} cannot get bytes")

    def get_bits(self):
        """Return self.data as bits"""
        if self.is_bit(False):
            return self._data
        elif self.is_byte(False):
            return np.unpackbits(self._data)
        else:
            raise ValueError(f"({self.PP}/{self.DT}) {self._data.dtype} cannot get bits")

    @classmethod
    def from_bytes(cls, data, pp=pproc.UNK, dt=dtype.BYTES):
        """Create Digital from bytes"""
        if isinstance(data, bytes):
            return cls(np.frombuffer(data, dtype=dtype.BYTES.dtype), pp, dt)
        elif isinstance(data, np.ndarray):
            if data.dtype == dtype.BYTES.dtype:
                return cls(data, pp, dt)
            raise NotImplementedError(f"Cannot convert {type(data)} {data.shape} {data.dtype} to {cls.__name__}")
        raise NotImplementedError(f"Cannot convert {type(data)} to {cls.__name__}")

    @classmethod
    def from_bits(cls, data, pp=pproc.UNK, dt=dtype.BITS):
        """Create Digital from bits"""
        if isinstance(data, np.ndarray):
            if data.dtype == dtype.BITS.dtype:
                return cls(data, pp, dt)
            raise NotImplementedError(f"Cannot convert {type(data)} {data.shape} {data.dtype} to {cls.__name__}")
        raise NotImplementedError(f"Cannot convert {type(data)} to {cls.__name__}")

def Data(data) -> Digital:
    """Wrapper to get correct Data type, and set PP and DT"""
    # Setting this to Digital hides errors on client side
    d = None
    if isinstance(data, Buffer):
        data = data._stor
    if isinstance(data, np.ndarray):
        if data.dtype == bool:
            d = Digital(data, dt=dtype.BITS)
        elif data.dtype == np.uint8:
            d = Digital(data, dt=dtype.BYTES)
        elif data.dtype == dtype.SAMPLES.dtype:
            d = Analog(data, dt=dtype.SAMPLES)
        elif data.dtype == dtype.SYMBOLS.dtype:
            d = Analog(data, dt=dtype.SYMBOLS)
        else:
            raise NotImplementedError(f"Cannot convert {type(data)} {data.shape} {data.dtype}")
    elif isinstance(data, str):
        d = Digital.from_bytes(data.encode("utf-8"), dt=dtype.BYTES)
    elif isinstance(data, bytes):
        d = Digital.from_bytes(data, dt=dtype.BYTES)

    if d is None:
        raise NotImplementedError(f"Cannot convert {type(data)}")
    # print(f"Converted {type(data)} to {d}")
    return d # type: ignore
