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
from .rpps import FLUSH

class _Data:
    __slots__ = (
        "_data",
        "PP", "DT", "ctx"
    )

    def __init__(self, data=None, pp=pproc.UNK, dt=None, ctx=None):
        self._data: np.ndarray = data # type: ignore
        self.PP = pp
        self.DT = dt
        self.ctx = {} if ctx is None else ctx # Stream state the stages carry between calls (see `derive`)

    @property
    def flush(self):
        """Is this the end of the stream? Stages hold a partial block for the next data unless it is: then a
        partial last block is padded (encoding) or dropped (decoding), and the stages finish (a terminated conv code's tail)"""
        return bool(self.ctx.get(FLUSH, False))

    @flush.setter
    def flush(self, value):
        self.ctx[FLUSH] = bool(value)

    def derive(self, data, pp, dt, cls=None, ctx=None):
        """New data of this stream: the stages never modify their input. `ctx` is the stream state they updated"""
        return (cls or type(self))(data, pp, dt, dict(self.ctx) if ctx is None else ctx)

    def __str__(self):
        return f"{self.name} ({self.PP}/{self.DT}) {len(self)}"

    def __len__(self):
        if self._data is None:
            return 0
        return len(self._data)

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
        """Create a copy of data (the stream state is copied too)"""
        return type(self)(self._data, self.PP, self.DT, dict(self.ctx))

class Analog(_Data):
    """Analog data"""
    pass

class Mod(_Data):
    """(De)Modulation data

    Modulated: complex symbols (pp MOD).
    Demodulated (pp MAP): either hard bits (MAPPED_HARD), or one LLR per bit (LLR). An LLR is
    log(P(bit=0)/P(bit=1)), so positive means 0 and the magnitude is the confidence.
    """
    def __init__(self, data=None, pp=pproc.MAP, dt=None, ctx=None):
        if pp == pproc.MAP and dt is None:
            dt = dtype.LLR if np.asarray(data).dtype.kind == "f" else dtype.MAPPED_HARD
        elif pp not in (pproc.MOD, pproc.MAP):
            raise ValueError(f"Cannot initialize Mod from {pp}")
        super().__init__(data, pp, dt, ctx)

    def bits(self):
        """Hard decided bits, leaving this data as it is"""
        if self.PP != pproc.MAP:
            raise ValueError(f"Cannot get bits of Mod {self.PP}/{self.DT}!")
        return self._data < 0 if self.DT == dtype.LLR else self._data.astype(bool)

    @property
    def is_soft(self):
        """Does this carry LLRs (rather than hard bits)"""
        return self.DT == dtype.LLR

    def get_soft(self, hard_llr=None):
        """Per-bit LLRs. Hard bits give +-hard_llr if it's set, otherwise None"""
        if self.is_soft:
            return self._data
        if hard_llr is not None and self.DT == dtype.MAPPED_HARD:
            return hard_llr * (1.0 - 2.0*self._data.astype(np.float32))
        return None

    def to_analog(self):
        """Convert to Analog"""
        if self.PP == pproc.MOD:
            return Analog(self._data, self.PP, self.DT, self.ctx)
        raise ValueError(f"Cannot convert Mod {self.PP}/{self.DT} to Analog!")

    def to_digital(self):
        """Convert to Digital"""
        if self.PP == pproc.MAP:
            self.as_hard()
            return Digital(self._data, pp=pproc.MAP, dt=dtype.BITS, ctx=self.ctx)
        raise ValueError(f"Cannot convert Mod {self.PP}/{self.DT} to Digital!")

    def as_hard(self):
        """Make hard decisions: bit 1 where the LLR is negative"""
        if self.DT == dtype.LLR:
            self._data = self._data < 0
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


    def bits(self):
        """The data as bits, leaving this data as it is"""
        return self.get_bits()

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
    def from_bytes(cls, data, pp=pproc.UNK, dt=dtype.BYTES, ctx=None):
        """Create Digital from bytes"""
        if isinstance(data, bytes):
            return cls(np.frombuffer(data, dtype=dtype.BYTES.dtype), pp, dt, ctx)
        elif isinstance(data, np.ndarray):
            if data.dtype == dtype.BYTES.dtype:
                return cls(data, pp, dt, ctx)
            raise NotImplementedError(f"Cannot convert {type(data)} {data.shape} {data.dtype} to {cls.__name__}")
        raise NotImplementedError(f"Cannot convert {type(data)} to {cls.__name__}")

    @classmethod
    def from_bits(cls, data, pp=pproc.UNK, dt=dtype.BITS, ctx=None):
        """Create Digital from bits"""
        if isinstance(data, np.ndarray):
            if data.dtype == dtype.BITS.dtype:
                return cls(data, pp, dt, ctx)
            raise NotImplementedError(f"Cannot convert {type(data)} {data.shape} {data.dtype} to {cls.__name__}")
        raise NotImplementedError(f"Cannot convert {type(data)} to {cls.__name__}")

def Data(data, ctx=None, flush=False) -> Digital:
    """Wrapper to get correct Data type, and set PP and DT

    ctx: the stream state of earlier data (`out.ctx`) to continue a stream in chunks
    flush: this ends the stream. Otherwise a partial last block is held for the next data; flushed, it is padded (encode) or dropped (decode)
    """
    # Setting this to Digital hides errors on client side
    d = None
    ctx = dict(ctx) if ctx else {}
    ctx[FLUSH] = bool(flush)
    if isinstance(data, Buffer):
        data = data._stor
    if isinstance(data, np.ndarray):
        if data.dtype == bool:
            d = Digital(data, dt=dtype.BITS, ctx=ctx)
        elif data.dtype == np.uint8:
            d = Digital(data, dt=dtype.BYTES, ctx=ctx)
        elif data.dtype == dtype.SAMPLES.dtype:
            d = Analog(data, dt=dtype.SAMPLES, ctx=ctx)
        elif data.dtype == dtype.SYMBOLS.dtype:
            d = Analog(data, dt=dtype.SYMBOLS, ctx=ctx)
        else:
            raise NotImplementedError(f"Cannot convert {type(data)} {data.shape} {data.dtype}")
    elif isinstance(data, str):
        d = Digital.from_bytes(data.encode("utf-8"), dt=dtype.BYTES, ctx=ctx)
    elif isinstance(data, bytes):
        d = Digital.from_bytes(data, dt=dtype.BYTES, ctx=ctx)

    if d is None:
        raise NotImplementedError(f"Cannot convert {type(data)}")
    return d # type: ignore
