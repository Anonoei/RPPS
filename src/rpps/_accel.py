"""The optional C++ kernels (rpps._native), or None

Stages look `native` up when they run, so it can be switched off at any time:

    RPPS_NO_NATIVE=1 python ...        # never load the kernels
    rpps._accel.native = None          # use the Python loops from now on (tests compare the two)

Every kernel has a Python reference loop in the stage that calls it; the two agree to rounding.
"""
import os

native = None
if not os.environ.get("RPPS_NO_NATIVE"):
    try:
        from . import _native as native
    except ImportError:
        native = None

def stale(native, src_dir):
    """Why `native` was not built from the C++ sources in src_dir, or None when it was (or there are no sources: an
    installed package)"""
    if native is None or not src_dir.is_dir():
        return None
    from ._srchash import source_hash
    built = getattr(native, "SRC_HASH", None)
    if built is None:
        return "it predates source hashes"
    if built != source_hash(src_dir):
        return "csrc/rpps_native changed since it was built"
    return None

def _check_checkout():
    """In a source checkout (csrc/ next to src/), warn when the in-place build is older than the C++ sources: the
    kernels would silently run old code"""
    import warnings
    from pathlib import Path
    why = stale(native, Path(__file__).resolve().parents[2] / "csrc" / "rpps_native")
    if why:
        warnings.warn(f"rpps._native is stale ({why}): rebuild it with `python tools/builder.py -c`", RuntimeWarning)

_check_checkout()

# What the per-sample kernels (squelch, dc_block, one_pole, the demodulators) take: anything else runs the Python loop
NATIVE_DTYPES = ("float32", "float64", "complex64", "complex128")

def available():
    """True when the C++ kernels are loaded and enabled"""
    return native is not None
