"""Boolean masks along one axis"""
import numpy as np

def runs(mask):
    """(starts, stops) of the True runs of a 1-D mask, stops exclusive: two int arrays of the same length"""
    m = np.asarray(mask)
    d = np.diff(np.concatenate(([0], m.view(np.int8) if m.dtype == bool else (m != 0).astype(np.int8), [0])))
    return np.flatnonzero(d == 1), np.flatnonzero(d == -1)
