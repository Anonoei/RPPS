"""Data containers

Every block carries three tags (see rpps.base.types and ARCHITECTURE.md):
    CAT  the Category of the operation that made it last
    REP  its representation: SAMPLES, SYMBOLS, CHANNEL bits, DATA bits, PACKETS
    DT   its form (dtype): complex samples, LLRs, bits, bytes...

    Analog   samples and symbols
    Mod      symbols, and demodulated bits (LLRs or hard) on the way to the decoder
    Digital  bits and bytes
"""

from time import perf_counter

import numpy as np

from .types import Category, Rep, dtype, REP_OF
from .buffer import Buffer
from .meta import Meta
from .rpps import FLUSH
from ..errors import StreamError

class _Data:
    __slots__ = (
        "_data",
        "CAT", "REP", "DT", "ctx", "meta"
    )

    def __init__(self, data=None, cat=Category.UNK, dt=None, ctx=None, meta=None, rep=None):
        self._data: np.ndarray = data # type: ignore
        self.CAT = cat
        self.DT = dt
        self.REP = rep if rep is not None else REP_OF.get(dt) # Inferred from the form unless the maker says
        self.ctx = {} if ctx is None else ctx # State the coding/scrambling pipes carry between calls (see `derive`)
        self.meta: Meta = Meta() if meta is None else meta # Rates and the stream of the signal

    @property
    def flush(self):
        """Is this the end of the stream? Stages hold a partial block for the next data unless it is: then a
        partial last block is padded (encoding) or dropped (decoding), and the stages finish (a terminated conv code's tail)"""
        return bool(self.ctx.get(FLUSH, False))

    @flush.setter
    def flush(self, value):
        self.ctx[FLUSH] = bool(value)

    def derive(self, data, cat, dt, cls=None, ctx=None, meta=None, rep=None):
        """New data of this stream: the stages never modify their input. `ctx` is the stream state they updated.
        rep: the representation of the new data; by default the one its form usually has (REP_OF): bits are data bits,
        LLRs channel bits. A code transform says which side of the decoder its bits are on"""
        return (cls or type(self))(data, cat, dt, dict(self.ctx) if ctx is None else ctx, self.meta if meta is None else meta, rep)

    # --- signal information ---
    @property
    def Fs(self):
        """Sample rate [Hz]"""
        return self.meta.Fs
    @property
    def sps(self):
        """Samples per symbol"""
        return self.meta.sps
    @property
    def cf(self):
        """Centre frequency [Hz]"""
        return self.meta.cf
    @property
    def Ts(self):
        return self.meta.Ts
    @property
    def delay(self):
        """Group delay of the stages so far [s]"""
        return self.meta.delay
    @property
    def det(self):
        """What `rp.sync.Acquire` detected (baud, modulation, cfo...), None before"""
        return self.meta.stream.det
    @property
    def stream(self):
        return self.meta.stream

    def state(self, stage):
        """The carried state of a stage in this stream (loop estimates, filter history...), None before it ran"""
        return self.meta.stream.states.get(stage)

    def apply(self, stage):
        """Run a DSP stage on this block: `data + stage`. The stage's state lives in the stream, the rates follow"""
        takes = stage.takes(True) if hasattr(stage, "takes") else {Rep.SAMPLES, Rep.SYMBOLS}
        if self._data is None or (takes is not None and self.REP not in takes):
            names = ", ".join(sorted(r.name for r in takes)) if takes else "nothing"
            raise StreamError(f"{stage} processes {names}, not {self.REP} ({self.name}/{self.DT})")
        meta = self.meta.fork()
        meta.refresh()
        meta.flush = self.flush
        st = meta.state_of(stage)
        t = perf_counter()
        out = stage.step(self._data, st, meta)
        stream = meta.stream
        busy = stream.busy
        busy[stage] = busy.get(stage, 0.0) + perf_counter() - t
        out = np.asarray(out)
        if out.dtype in (np.complex128, np.float64): # Samples are single precision throughout
            out = out.astype(np.complex64 if out.dtype.kind == "c" else np.float32)
        cat = getattr(stage, "CATEGORY", None) or Category.SIGNAL
        if self.REP in (Rep.CHANNEL, Rep.DATA): # A stage on bits: same class, representation and form
            out_meta = meta.fork()
            start = stream.out_pos.get(stage)
            out_meta.pos = meta.pos if start is None else start
            stream.out_pos[stage] = out_meta.pos + len(out)
            return self.derive(out, cat, self.DT, ctx=dict(self.ctx), meta=out_meta, rep=self.REP)
        if out.dtype.kind == "c":
            # One value per symbol is symbols: a stage that takes samples to one per symbol (timing recovery, an FFT)
            # gives them, and a stage on symbols keeps them
            dt = dtype.SYMBOLS if meta.op.sps == 1 or (meta.op.sps <= 1.0 and self.DT == dtype.SYMBOLS) else dtype.SAMPLES
        else:
            dt = dtype.REAL
        out_meta = meta.fork()
        start = stream.out_pos.get(stage)
        if start is None: # First block this stage sees in the stream: its outputs start where its input is, at the new rate
            start = int(round(meta.pos * meta.op.Fs / meta.state.Fs)) if meta.state.Fs else 0
        out_meta.pos = start # Then it counts its own outputs: independent of how the blocks were cut
        stream.out_pos[stage] = start + len(out)
        return Analog(out, cat, dt, dict(self.ctx), out_meta)

    def __rshift__(self, sink):
        """data >> sink: write this block to a sink"""
        return sink.write(self)

    def __str__(self):
        return f"{self.name} ({self.REP}/{self.DT}) {len(self)}"

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
        return type(self)(self._data, self.CAT, self.DT, dict(self.ctx), self.meta, self.REP)

class Analog(_Data):
    """Analog data"""

class Mod(_Data):
    """(De)Modulation data

    Symbols (rep SYMBOLS), or demodulated bits (rep CHANNEL, or DATA from a soft-output decoder): either hard bits
    (MAPPED_HARD), or one LLR per bit (LLR). An LLR is log(P(bit=0)/P(bit=1)), so positive means 0 and the magnitude
    is the confidence.
    """
    def __init__(self, data=None, cat=Category.SYMBOL, dt=None, ctx=None, meta=None, rep=None):
        if dt is None:
            dt = dtype.LLR if np.asarray(data).dtype.kind == "f" else dtype.MAPPED_HARD
        super().__init__(data, cat, dt, ctx, meta, rep)

    def bits(self):
        """Hard decided bits, leaving this data as it is"""
        if self.REP not in (Rep.CHANNEL, Rep.DATA):
            raise ValueError(f"Cannot get bits of Mod {self.REP}/{self.DT}!")
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
        if self.REP == Rep.SYMBOLS:
            return Analog(self._data, self.CAT, self.DT, self.ctx, self.meta, self.REP)
        raise ValueError(f"Cannot convert Mod {self.REP}/{self.DT} to Analog!")

    def to_digital(self):
        """Convert to Digital"""
        if self.REP in (Rep.CHANNEL, Rep.DATA):
            self.as_hard()
            return Digital(self._data, self.CAT, dtype.BITS, self.ctx, self.meta, self.REP)
        raise ValueError(f"Cannot convert Mod {self.REP}/{self.DT} to Digital!")

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
        raise ValueError(f"({self.REP}/{self.DT}) {self._data.dtype} cannot be shown in binary")

    @property
    def hex(self):
        """Return hexadecimal implementation"""
        if self.is_byte():
            return self._data.tobytes().hex()
        raise ValueError(f"({self.REP}/{self.DT}) {self._data.dtype} cannot be shown in hex")

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
            raise ValueError(f"({self.REP}/{self.DT}) {self._data.dtype} cannot get bytes")

    def get_bits(self):
        """Return self.data as bits"""
        if self.is_bit(False):
            return self._data
        elif self.is_byte(False):
            return np.unpackbits(self._data)
        else:
            raise ValueError(f"({self.REP}/{self.DT}) {self._data.dtype} cannot get bits")

    @classmethod
    def from_bytes(cls, data, cat=Category.UNK, dt=dtype.BYTES, ctx=None, meta=None, rep=None):
        """Create Digital from bytes"""
        if isinstance(data, bytes):
            return cls(np.frombuffer(data, dtype=dtype.BYTES.dtype), cat, dt, ctx, meta, rep)
        elif isinstance(data, np.ndarray):
            if data.dtype == dtype.BYTES.dtype:
                return cls(data, cat, dt, ctx, meta, rep)
            raise TypeError(f"Cannot convert {type(data)} {data.shape} {data.dtype} to {cls.__name__}")
        raise TypeError(f"Cannot convert {type(data)} to {cls.__name__}")

    @classmethod
    def from_bits(cls, data, cat=Category.UNK, dt=dtype.BITS, ctx=None, meta=None, rep=None):
        """Create Digital from bits"""
        if isinstance(data, np.ndarray):
            if data.dtype == dtype.BITS.dtype:
                return cls(data, cat, dt, ctx, meta, rep)
            raise TypeError(f"Cannot convert {type(data)} {data.shape} {data.dtype} to {cls.__name__}")
        raise TypeError(f"Cannot convert {type(data)} to {cls.__name__}")

def Data(data, ctx=None, flush=False, Fs=None, sps=None, cf=None, stream=None, delay=0.0, t0=None) -> Digital | Analog:
    """Wrapper to get correct Data type, and set PP and DT

    bool -> bits, uint8/bytes/str -> bytes, complex64 -> samples, float -> real samples
    ctx: the state of earlier data (`out.ctx`) to continue a stream of bits in chunks
    flush: this ends the stream. Otherwise a partial last block is held for the next data; flushed, it is padded (encode) or dropped (decode)
    Fs, sps, cf: sample rate [Hz], samples per symbol and centre frequency [Hz] of a signal
    stream: the stream of earlier signal blocks (`out.stream`), which keeps the state of the stages across blocks
    t0: time [s] of the stream's first item, if known (a radio's timestamp): `meta.time_of` then gives any item's time
    Each block gets its absolute position in the stream (`meta.pos`): the items its stream produced before it
    """
    d = None
    ctx = dict(ctx) if ctx else {}
    ctx[FLUSH] = bool(flush)
    meta = Meta(1.0 if Fs is None else Fs, 0.0 if sps is None else sps, 0.0 if cf is None else cf, delay, stream)
    if t0 is not None:
        meta.stream.t0 = float(t0)
    if isinstance(data, Buffer):
        data = data._stor
    if isinstance(data, np.ndarray):
        if data.dtype == bool:
            d = Digital(data, dt=dtype.BITS, ctx=ctx, meta=meta)
        elif data.dtype == np.uint8:
            d = Digital(data, dt=dtype.BYTES, ctx=ctx, meta=meta)
        elif data.dtype == dtype.SAMPLES.dtype:
            d = Analog(data, dt=dtype.SYMBOLS if meta.state.sps == 1 else dtype.SAMPLES, ctx=ctx, meta=meta)
        elif data.dtype == np.complex128:
            d = Analog(data.astype(np.complex64), dt=dtype.SAMPLES, ctx=ctx, meta=meta)
        elif data.dtype.kind == "f":
            d = Analog(data, dt=dtype.REAL, ctx=ctx, meta=meta)
        else:
            raise TypeError(f"Cannot convert {type(data)} {data.shape} {data.dtype}")
    elif isinstance(data, str):
        d = Digital.from_bytes(data.encode("utf-8"), dt=dtype.BYTES, ctx=ctx, meta=meta)
    elif isinstance(data, bytes):
        d = Digital.from_bytes(data, dt=dtype.BYTES, ctx=ctx, meta=meta)

    if d is None:
        raise TypeError(f"Cannot convert {type(data)}")
    d.meta.pos = d.meta.stream.pos # The stream's source position: this block follows what it produced before
    d.meta.stream.pos += len(d)
    return d # type: ignore

def expect(op, data, takes, encode):
    """Refuse `data` an operator does not take: `takes` is the set of representations it accepts (None: anything).
    Raised before any work, with what was expected"""
    if takes is None or not isinstance(data, _Data):
        return
    if data.REP not in takes:
        names = ", ".join(sorted(r.name for r in takes))
        raise StreamError(f"{op} {'encodes' if encode else 'decodes'} {names}, not {data.REP} ({data.name}/{data.DT})")
