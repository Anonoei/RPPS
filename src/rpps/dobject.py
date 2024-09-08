"""Containers for data to ensure we can track where it came from"""

from enum import Enum
import numpy as np

from .base.soft import SoftDecision
from .meta import Meta

class Type(Enum):
    """DataObject data data formats"""
    BIT = 0
    BYTE = 1
    SYM = 2
    NONE = -1
def ensure_bit(dobj):
    """Ensure DataObject is using bits"""
    return BitObject(dobj, dobj.meta)

def ensure_byte(dobj):
    """Ensure DataObject is using bytes"""
    return ByteObject(dobj, dobj.meta)

class DataObject:
    """Parent DataObject class"""
    type = Type.NONE

    data = []
    meta = None

    def __init__(self, data=None, meta=None):
        if not isinstance(meta, Meta):
            meta = Meta()
        self.convert(data)
        self.meta = meta

    def __str__(self):
        return f"{self.name()}:{self.type}:{len(self)}"

    def __repr__(self):
        return str(self)

    def __len__(self):
        return len(self.data)

    def __enter__(self, *args, **kwargs):
        return self.data.__enter__(*args, **kwargs) # type: ignore

    def __exit__(self, *args, **kwargs):
        return self.data.__exit__(*args, **kwargs) # type: ignore

    def __iter__(self, *args, **kwargs):
        return self.data.__iter__(*args, **kwargs)

    def __getitem__(self, *args, **kwargs):
        return self.data.__getitem__(*args, **kwargs)

    def name(self):
        """Get class name"""
        return type(self).__name__

    def append(self, data):
        self.data = np.append(self.data, data)

    def to_bits(self):
        if isinstance(self.data, np.ndarray):
            if self.type == Type.BIT:
                return self.data
            elif self.type == Type.BYTE:
                return np.unpackbits(self.data)
        raise NotImplementedError(f"Cannot convert {self.type} to bits")

    def to_bytes(self):
        if isinstance(self.data, np.ndarray):
            if self.type == Type.BYTE:
                return self.data
            if self.type == Type.BIT:
                return np.packbits(self.data)
        raise NotImplementedError(f"Cannot convert {self.type} to bytes")

    def convert(self, other):
        """Convert other to self"""
        if issubclass(type(other), DataObject):
            if self.type == Type.BIT:
                self.data = other.to_bits()
                return
            elif self.type == Type.BYTE:
                self.data = other.to_bytes()
                return
            raise NotImplementedError(f"Cannot convert {other.type} to {self.type}")
        else:
            if other is None:
                if self.type == Type.BYTE:
                    self.data = np.array([], dtype=np.uint8)
                    return
                elif self.type == Type.BIT:
                    self.data = np.array([], dtype=bool)
                    return
            elif isinstance(other, bytes):
                if self.type == Type.BYTE:
                    self.data = np.frombuffer(other, dtype=np.uint8)
                    return
                elif self.type == Type.BIT:
                    self.data = np.unpackbits(np.array(other))
                    return
            elif isinstance(other, np.ndarray):
                if self.type == Type.SYM:
                    self.data = other
                    return
                if other.dtype == np.uint8:
                    if self.type == Type.BYTE:
                        self.data = np.copy(other)
                        return
                    elif self.type == Type.BIT:
                        self.data = np.unpackbits(other)
                        return
                elif other.dtype == bool:
                    if self.type == Type.BYTE:
                        self.data = np.packbits(other)
                        return
                    elif self.type == Type.BIT:
                        self.data = np.copy(other)
                        return
                raise NotImplementedError(f"Cannot convert {type(other)} {other.shape} {other.dtype} to {self.type}")

        raise NotImplementedError(f"Cannot convert {type(other)} to {self.type}")


# --- Base types
class BitObject(DataObject):
    type = Type.BIT
    data = np.array([], dtype=bool)

    def __str__(self):
        return f"{super().__str__()}:{self.bin}"

    @property
    def bin(self):
        return self.data.astype(int)


class ByteObject(DataObject):
    type = Type.BYTE
    data = np.array([], dtype=np.uint8)

    def __str__(self):
        return f"{super().__str__()}:{self.hex}"

    @property
    def hex(self):
        return self.data.tobytes().hex()


class SymObject(DataObject):
    type = Type.SYM

    def __str__(self):
        return f"{super().__str__()} syms"


# --- Implementations
class StreamData(ByteObject):
    """Raw input/output data"""


# --- Scram
class ScramData(BitObject):
    """Data post-scrambler"""


# --- Coding
class CodingData(BitObject):
    """Data post-coding"""


# --- Mod
class ModData(BitObject):
    """Data returned from mod.demodulate"""
    soft = SoftDecision()
    _data = None

    @property
    def hard(self):
        return self.soft.hard()

    @property
    def data(self):
        if self._data is not None:
            return self._data
        return self.soft.bits

    @data.setter
    def data(self, value):
        self._data = value


class SymData(SymObject):
    """Data returned from mod.modulate"""
