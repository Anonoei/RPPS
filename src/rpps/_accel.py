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

# What the per-sample kernels (squelch, dc_block, one_pole, the demodulators) take: anything else runs the Python loop
NATIVE_DTYPES = ("float32", "float64", "complex64", "complex128")

def available():
    """True when the C++ kernels are loaded and enabled"""
    return native is not None
