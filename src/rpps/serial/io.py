from io import BytesIO
from io import BufferedReader, BufferedWriter

import numpy as np

class Format:
    SIZE = 0
    @classmethod
    def read(cls, io, count):
        raise NotImplementedError()
    @classmethod
    def write(cls, io, data):
        io.write(data.tobytes())

    @classmethod
    def bits(cls):
        """Return size of format in bits"""
        return cls.SIZE

    @classmethod
    def bytes(cls):
        """Return size of format in bytes"""
        return cls.SIZE // 8

    @classmethod
    def _read(cls, io, count, dtype):
        return np.frombuffer(io.read(int(cls.bytes()*count)), dtype=dtype)

# --- Integers --- #
class i8(Format):
    """Eq. to C int8_t"""
    SIZE = 8
    @classmethod
    def read(cls, io, count):
        return cls._read(io, count, dtype=np.int8)

class i16(Format):
    """Eq. to C int16_t"""
    SIZE = 16
    @classmethod
    def read(cls, io, count):
        return cls._read(io, count, dtype=np.int16)

class i32(Format):
    """Eq. to C int32_t"""
    SIZE = 32
    @classmethod
    def read(cls, io, count):
        return cls._read(io, count, dtype=np.int32)


class ui8(Format):
    """Eq. to C uint8_t"""
    SIZE = 8
    @classmethod
    def read(cls, io, count):
        return cls._read(io, count, dtype=np.uint8)

class ui16(Format):
    """Eq. to C uint16_t"""
    SIZE = 8
    @classmethod
    def read(cls, io, count):
        return cls._read(io, count, dtype=np.uint16)

class ui32(Format):
    """Eq. to C int32_t"""
    SIZE = 8
    @classmethod
    def read(cls, io, count):
        return cls._read(io, count, dtype=np.uint32)

class i4(Format):
    """4 bit integer"""
    SIZE = 4
    @classmethod
    def read(cls, io, count):
        buff = np.empty(count, dtype=np.int8)
        _bytes = np.frombuffer(io, dtype=np.uint8, count=count//2)
        buff[::2] = (_bytes & 0b11110000) >> 4
        buff[1::2] = (_bytes & 0b00001111)
        return buff

# --- Floats --- #
class f16(Format):
    """Eq. to C float16_t"""
    SIZE = 16
    @classmethod
    def read(cls, io, count):
        return cls._read(io, count, dtype=np.float16)

class f32(Format):
    """Eq. to C float32_t"""
    SIZE = 32
    @classmethod
    def read(cls, io, count):
        return cls._read(io, count, dtype=np.float32)

class f64(Format):
    """Eq. to C float64_t"""
    SIZE = 32
    @classmethod
    def read(cls, io, count):
        return cls._read(io, count, dtype=np.float64)

# --- Complex --- #
class ComplexFormat(Format):
    """Format subclass for complex numbers"""
    BASE: type = None # type: ignore
    @classmethod
    def read(cls, io, count):
        raise NotImplementedError()

class ci8(ComplexFormat):
    """Complex int8 (i4, i4)"""
    SIZE = 8
    BASE = i4
    @classmethod
    def read(cls, io, count):
        return cls.BASE.read(io, count*2).astype(np.float32).view(dtype=np.complex64)

class ci16(ComplexFormat):
    """Complex int16 (i8, i8)"""
    SIZE = 16
    BASE = i8
    @classmethod
    def read(cls, io, count):
        return cls.BASE.read(io, count*2).astype(np.float32).view(dtype=np.complex64)
class ci32(ComplexFormat):
    """Complex int32 (i16, i16)"""
    SIZE = 32
    BASE = i16
    @classmethod
    def read(cls, io, count):
        return cls.BASE.read(io, count*2).astype(np.float32).view(dtype=np.complex64)

class cf32(ComplexFormat):
    """Complex float32 (f16, f16)"""
    SIZE = 32
    BASE = f16
    @classmethod
    def read(cls, io, count):
        return cls.BASE.read(io, count*2).astype(np.float32).view(dtype=np.complex64)

class cf64(ComplexFormat):
    """Complex float64 (f32, f32)"""
    SIZE = 64
    BASE = f32
    @classmethod
    def read(cls, io, count):
        return cls.BASE.read(io, count*2).view(dtype=np.complex64)

class cf128(ComplexFormat):
    """Complex float128 (f64, f64)"""
    SIZE = 128
    BASE = f64
    @classmethod
    def read(cls, io, count):
        return cls.BASE.read(io, count*2).view(dtype=np.complex128)
