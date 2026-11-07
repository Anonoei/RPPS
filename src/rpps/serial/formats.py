"""Sample file formats, as pysdrlib names them: a complex format is named by the bits of one whole sample

    ci8t ci16t ci32t   complex two's compliment integers (4, 8 and 16 bit I and Q)
    cf32 cf64 cf128    complex floats (cf64 is two float32: numpy's complex64)
    i8t.. ui8.. f32..  real samples
`rp.serial.formats.ls()` lists them.
"""
import numpy as np
import pysdrlib
from pysdrlib.base import io as _io

Formats = pysdrlib.Formats

def get(fmt):
    """A pysdrlib Formats member from its name (or a member)"""
    if isinstance(fmt, Formats):
        return fmt
    try:
        return Formats[fmt]
    except KeyError:
        raise ValueError(f"Unknown format {fmt!r}, have {ls()}") from None

def ls():
    """Names that can be used for a format"""
    return list(Formats.__members__)

def full_scale(fmt):
    """Value that integer samples of this format map to 1.0 with (1.0 for floats)"""
    cls = get(fmt).value
    base = getattr(cls, "BASE", cls)
    while getattr(base, "BASE", None) is not None:
        base = base.BASE
    if base.DTYPE is not None and np.dtype(base.DTYPE).kind == "f":
        return 1.0
    return float(2**(base.SIZE - 1))
