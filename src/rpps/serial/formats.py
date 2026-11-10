"""Sample file formats, as pysdrlib names them: a complex format is named by the bits of one whole sample

    ci8t ci16t ci32t   complex two's compliment integers (4, 8 and 16 bit I and Q)
    cf32 cf64 cf128    complex floats (cf64 is two float32: numpy's complex64)
    i8t.. ui8.. f32..  real samples
`rp.serial.formats.ls()` lists them.
"""
from pysdrlib import Formats

ls = Formats.ls

def get(fmt):
    """A pysdrlib Formats member from its name (or a member)"""
    if isinstance(fmt, Formats):
        return fmt
    try:
        return Formats[fmt]
    except KeyError:
        raise ValueError(f"Unknown format {fmt!r}, have {ls()}") from None
