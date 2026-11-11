"""FFT convolution for long filters"""
import numpy as np

_DIRECT = 16        # Filters shorter than this are cheaper to convolve directly
_MIN_SIGNAL = 4096  # and so are short signals

def convolve(x, h, mode="full"):
    """Convolution of the signal `x` with the filter `h` by overlap-add FFT blocks when that is cheaper than `np.convolve`

    mode: "full" (len(x) + len(h) - 1 samples), "same" (len(x), centred on the full output) or "valid" (len(x) - len(h) + 1, no edges)
    Real and complex inputs; the result is float32 / complex64 when both inputs are single precision, else double.
    """
    if mode not in ("full", "same", "valid"):
        raise ValueError(f"mode must be full, same or valid, not {mode!r}")
    x = np.asarray(x); h = np.asarray(h)
    if x.ndim != 1 or h.ndim != 1:
        raise ValueError("convolve takes 1-D signals")
    if len(x) < len(h) and mode == "valid": # Same as np.convolve: the shorter one is the filter
        x, h = h, x
    nx, nh = len(x), len(h)
    single = x.dtype in (np.float32, np.complex64) and h.dtype in (np.float32, np.complex64, np.float16)
    if nh < _DIRECT or nx < _MIN_SIGNAL or nh*8 > nx:
        out = np.convolve(x, h, "full")
    else:
        cplx = np.iscomplexobj(x) or np.iscomplexobj(h)
        n = 1 << int(np.ceil(np.log2(8*nh)))        # FFT size: blocks of n - nh + 1 new samples
        step = n - nh + 1
        fwd, inv = (np.fft.fft, np.fft.ifft) if cplx else (np.fft.rfft, np.fft.irfft)
        kw = {} if cplx else {"n": n}
        H = fwd(h.astype(np.complex128 if cplx else np.float64), n)
        out = np.zeros(nx + nh - 1, dtype=np.complex128 if cplx else np.float64)
        xs = x.astype(out.dtype, copy=False)
        for a in range(0, nx, step):
            y = inv(fwd(xs[a:a + step], n)*H, **kw)
            m = min(n, len(out) - a)
            out[a:a + m] += y[:m]
    if mode == "same":
        s = (nh - 1)//2
        out = out[s:s + nx]
    elif mode == "valid":
        out = out[nh - 1:nx]
    dtype = (np.complex64 if np.iscomplexobj(out) else np.float32) if single else None
    return out.astype(dtype, copy=False) if dtype else out
