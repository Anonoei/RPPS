"""Sample file formats, as pysdrlib names them: a complex format is named by the bits of its I (or Q) alone, as in SigMF

    ci8t ci16t ci32t   complex two's complement integers (8, 16 and 32 bit I and Q)
    cf16 cf32 cf64     complex floats (cf32 is two float32: numpy's complex64; cf64 is complex128)
    i8t.. ui8.. f32..  real samples
`rp.source.formats.ls()` lists them.
"""
# pysdrlib is imported when a format is first asked for: `import rpps` works without it (codes, profiles, analysis)

NEED = (0, 5) # pysdrlib before 0.5 named formats otherwise ("cf32" was two float16): its files would be read as noise

def pysdrlib():
    """The pysdrlib module, refused (ImportError) when it is older than NEED"""
    import pysdrlib as lib
    have = tuple(int(p) for p in lib.__version__.split(".")[:2] if p.isdigit())
    if have < NEED:
        raise ImportError(f"rpps needs pysdrlib >= {'.'.join(map(str, NEED))}, this is {lib.__version__} ({lib.__file__}): "
                          f"its format names differ, so samples would be misread. pip install -U pysdrlib")
    return lib

def ls():
    """The format names pysdrlib knows"""
    return pysdrlib().Formats.ls()

def get(fmt):
    """A pysdrlib Formats member from its name (or a member)"""
    Formats = pysdrlib().Formats
    if isinstance(fmt, Formats):
        return fmt
    try:
        return Formats[fmt]
    except KeyError:
        raise ValueError(f"Unknown format {fmt!r}, have {ls()}") from None
