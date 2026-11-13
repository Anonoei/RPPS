"""Rolling statistics of a 1-D array"""
import numpy as np

_CHUNK = 1 << 23 # Elements of the window matrix built at a time

def percentile_filter(x, percentile, size, mode="nearest"):
    """The `percentile` (0-100) of every window of `size` samples centred on each sample of `x` (a rolling median at 50)

    The rank is int(size*percentile/100), the lowest sample being rank 0 (as scipy.ndimage.percentile_filter). The window reaches
    size//2 samples back and (size - 1)//2 ahead. mode: how it is filled past the ends: "nearest" (repeat the edge sample),
    "reflect", "mirror", "wrap", or "constant" (zeros)
    """
    x = np.asarray(x)
    size = int(size)
    if x.ndim != 1:
        raise ValueError("percentile_filter takes a 1-D array")
    if size < 1:
        raise ValueError("size must be at least 1")
    if not 0 <= percentile <= 100:
        raise ValueError("percentile must be within 0..100")
    if len(x) == 0:
        return x.copy()
    rank = min(int(size*percentile/100), size - 1)
    lo, hi = size//2, (size - 1)//2
    pad_mode = {"nearest": "edge", "reflect": "symmetric", "mirror": "reflect", "wrap": "wrap", "constant": "constant"}.get(mode)
    if pad_mode is None:
        raise ValueError(f"mode must be nearest, reflect, mirror, wrap or constant, not {mode!r}")
    xp = np.pad(x, (lo, hi), mode=pad_mode)
    win = np.lib.stride_tricks.sliding_window_view(xp, size)
    out = np.empty(len(x), dtype=x.dtype)
    step = max(_CHUNK//size, 1)
    for a in range(0, len(x), step):
        out[a:a + step] = np.partition(win[a:a + step], rank, axis=1)[:, rank]
    return out
