import numpy as np

from .vector import dot_vector

def _matrix_dot(x, y, psds, yt=0.05, yb=0.05):
    out = np.zeros((psds.shape[1], y, x), dtype=np.int8)

    amp_min, amp_max = np.min(psds), np.max(psds)
    amp_min = amp_min*(1+yb) if amp_min < 0 else amp_min*(1-yb)
    amp_max = amp_max*(1-yt) if amp_max < 0 else amp_max*(1+yt)

    amp_rng = abs(abs(amp_max) - abs(amp_min))
    amp_off = amp_min if amp_min > 0 else -amp_min

    x_ratio = x/psds.shape[0]
    y_ratio = y/amp_rng

    x_idx = np.floor(np.arange(0, psds.shape[0])*x_ratio).astype(int)
    y_idx = np.floor((psds+amp_off)*y_ratio).astype(int)

    for i in range(psds.shape[1]):
        matrix = np.zeros((y, x), dtype=np.int8)
        y_idx = np.floor((psds[:,i]+amp_off)*y_ratio).astype(int)
        matrix[y_idx, x_idx] = 1
        out[i] = matrix
    return out, (amp_min, amp_max)

def _merge(matrix):
    out = np.zeros((matrix.shape[1], matrix.shape[2]), dtype=np.int8)
    for i in range(matrix.shape[0]):
        out += matrix[i,:,:]
    return out

def sdot(x, y, psds, yt=0.05, yb=0.05):
    """(Slow) Dot Matrix
    Map each PSD to the matrix, then merge each matrix
    """
    mats, (amp_min, amp_max) = _matrix_dot(x, y, psds, yt=0.05, yb=0.05)
    return _merge(mats), (amp_min, amp_max)

def svec(x, y, psds, yt=0.05, yb=0.05):
    """(Slow) Vector Matrix
    On each matrix, insert 1s between x-1's y and x's y
    """
    mats, (amp_min, amp_max) = _matrix_dot(x, y, psds, yt=0.05, yb=0.05)
    for i in range(mats.shape[0]):
        mats[i] = dot_vector(mats[i])
    return _merge(mats), (amp_min, amp_max)

def dot(x, y, psds, yt=0.05, yb=0.05):
    """Dot Matrix
    Each PSD's x/y mapped to histogram x/y
    """
    hist = np.zeros((y, x), dtype=np.int8)

    amp_min, amp_max = np.min(psds), np.max(psds)
    amp_min = amp_min*(1+yb) if amp_min < 0 else amp_min*(1-yb)
    amp_max = amp_max*(1-yt) if amp_max < 0 else amp_max*(1+yt)

    amp_rng = abs(abs(amp_max) - abs(amp_min))
    amp_off = amp_min if amp_min > 0 else -amp_min

    x_ratio = x/psds.shape[0]
    y_ratio = y/amp_rng

    x_idx = np.floor(np.arange(0, psds.shape[0])*x_ratio).astype(int)
    y_idx = np.floor((psds+amp_off)*y_ratio).astype(int)

    for i in range(psds.shape[1]):
        hist[y_idx[:,i], x_idx] += 1
    return hist, (amp_min, amp_max)

def vec(x, y, psds, yt=0.05, yb=0.05, interp=32):
    """Vector Matrix
    Interpolate between PSD x/y to mimic vector matrix
    """
    y_int = np.zeros((psds.shape[0]*interp, psds.shape[1]))
    t = np.arange(psds.shape[0])*interp
    t_int = np.arange(psds.shape[0]*interp)-1
    for i in range(psds.shape[1]):
        y_int[:,i] = np.interp(t_int, t, psds[:,i])
    return dot(x, y, y_int, yt, yb)
