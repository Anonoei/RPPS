"""Interpolation"""
import numpy as np

def parabolic_offset(a, b, c):
    """Vertex offset (-0.5..0.5) of the parabola through three equally spaced samples a, b, c (b in the middle)"""
    den = a - 2*b + c
    return 0.0 if den == 0 else 0.5*(a - c)/den

def peak(spec, circular=False, log=True):
    """Index of the largest value of a spectrum, and the sub-bin offset from a parabola through its neighbours

    log: fit the parabola to log(spec), the right model for a spectral line
    circular: the spectrum wraps around (an FFT), so the neighbours of the end bins are the other end
    Returns (k, offset), the peak is at k + offset. A peak on the end bin of a non-circular spectrum gets offset 0.
    """
    spec = np.asarray(spec)
    k = int(np.argmax(spec))
    n = len(spec)
    if not circular and not 0 < k < n - 1:
        return k, 0.0
    a, b, c = spec[(k - 1) % n], spec[k], spec[(k + 1) % n]
    if log:
        a, b, c = np.log(a + 1e-30), np.log(b + 1e-30), np.log(c + 1e-30)
    return k, parabolic_offset(a, b, c)
