from enum import Enum

from . import io as IO

class Formats(Enum):
    """IO Format wrapper"""
    # Integer
    i8 = IO.i8
    i16 = IO.i16
    i32 = IO.i32
    ui8 = IO.ui8
    ui16 = IO.ui16
    ui32 = IO.ui32
    i4 = IO.i4
    # Float
    f16 = IO.f16
    f32 = IO.f32
    f64 = IO.f64
    # Complex
    ci8 = IO.ci8
    ci16 = IO.ci16
    ci32 = IO.ci32
    cf32 = IO.cf32
    cf64 = IO.cf64
    cf128 = IO.cf128

    def read(self, io, count):
        """Input as format"""
        return self.value.read(io, count)

    def write(self, io, data):
        """Output as format"""
        self.value.write(io, data)

    @property
    def bits(self):
        """Return size of format in bits"""
        return self.value.SIZE

    @property
    def bytes(self):
        """Return size of format in bytes"""
        return self.value.SIZE // 8
    @property
    def is_complex(self):
        if issubclass(self.value, IO.ComplexFormat):
            return True
        return False

    @classmethod
    def ls(cls):
        return [fmt.name for fmt in cls]

    @classmethod
    def get(cls, name: str):
        return cls[name]
