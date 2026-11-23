"""The optional C++ kernels (rpps._native), or None

Stages look `native` up when they run, so it can be switched off at any time:

    RPPS_NO_NATIVE=1 python ...        # never load the kernels
    rpps._accel.native = None          # use the Python loops from now on (tests compare the two)

Every kernel has a Python reference loop in the stage that calls it; the two agree to rounding.

`device()` is the array module of a GPU, for a parallel region's batch kernels (rpps.parallel): CuPy, which is built for
CUDA (NVIDIA) or for ROCm/HIP (AMD) with one API, so nothing here names a vendor. Nothing is chosen for you:

    xp, backend = rpps._accel.device()        # (cupy, "cuda" | "hip"), or (numpy, None) without a GPU or CuPy
    profile.rx(parallel=rp.parallel.Parallel("thread", xp=xp))
    RPPS_DEVICE=none | cuda | hip              # none: numpy; cuda / hip: that backend or an error
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

def device():
    """(xp, backend): CuPy and "cuda" or "hip" when it is installed and sees a GPU, else (numpy, None). RPPS_DEVICE=none forces
    numpy; RPPS_DEVICE=cuda or hip asks for that backend and raises RuntimeError when it is not what is there"""
    import numpy as np
    want = os.environ.get("RPPS_DEVICE", "").strip().lower()
    if want == "none":
        return np, None
    if want not in ("", "cuda", "hip"):
        raise RuntimeError(f"RPPS_DEVICE={want!r}: none, cuda or hip")
    try:
        import cupy
        backend = "hip" if bool(getattr(cupy.cuda.runtime, "is_hip", False)) else "cuda"
        if cupy.cuda.runtime.getDeviceCount() < 1:
            raise RuntimeError("no device")
    except Exception as e: # not installed, or no driver or device: numpy, unless a backend was asked for
        if want:
            raise RuntimeError(f"RPPS_DEVICE={want}: no usable GPU ({type(e).__name__}: {e})") from None
        return np, None
    if want and want != backend:
        raise RuntimeError(f"RPPS_DEVICE={want}, but CuPy is built for {backend}")
    return cupy, backend
