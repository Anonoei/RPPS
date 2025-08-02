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

def matrix_dot(x, y, psds, yt=0.05, yb=0.05):
    mats, (amp_min, amp_max) = _matrix_dot(x, y, psds, yt=0.05, yb=0.05)
    return _merge(mats), (amp_min, amp_max)

def matrix_vector(x, y, psds, yt=0.05, yb=0.05):
    mats, (amp_min, amp_max) = _matrix_dot(x, y, psds, yt=0.05, yb=0.05)
    for i in range(mats.shape[0]):
        mats[i] = dot_vector(mats[i])
    return _merge(mats), (amp_min, amp_max)

def matrix_fastdot(x, y, psds, yt=0.05, yb=0.05):
    hist = np.zeros((y, x), dtype=np.int8)

    amp_min, amp_max = np.min(psds), np.max(psds)
    amp_min = amp_min*(1+yb) if amp_min < 0 else amp_min*(1-yb)
    amp_max = amp_max*(1-yt) if amp_max < 0 else amp_max*(1+yt)

    # amp_rng = abs(amp_max) + abs(amp_min)
    amp_rng = abs(abs(amp_max) - abs(amp_min))
    amp_off = amp_min if amp_min > 0 else -amp_min

    x_ratio = x/psds.shape[0]
    y_ratio = y/amp_rng

    # print(f"amp_off: {amp_off:.2f}")

    # print(f"x ratio: {x_ratio:.2f}, y ratio: {y_ratio:.2f}")
    # print(f"X {psds.shape[0]} = hist {psds.shape[0]*x_ratio}")
    # print(f"Y {amp_min:.2f} = hist {(amp_min+amp_off)*y_ratio}")
    # print(f"Y {amp_max:.2f} = hist {(amp_max+amp_off)*y_ratio}")

    x_idx = np.floor(np.arange(0, psds.shape[0])*x_ratio).astype(int)
    y_idx = np.floor((psds+amp_off)*y_ratio).astype(int)

    for i in range(psds.shape[1]):
        hist[y_idx[:,i], x_idx] += 1
    # print(f"Histogram min/max/sum: {np.min(hist)}/{np.max(hist)}/{np.sum(hist)}")
    return hist, (amp_min, amp_max)
