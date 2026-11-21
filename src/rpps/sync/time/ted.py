"""Timing error detectors: the error a timing loop (rpps.sync.time.TLL) steers by, from samples around a symbol's instant

Both need no decisions and no carrier (non-data aided) and at least 2 samples a symbol. Positive: the instant is early (the
loop moves it later). Decision-directed Mueller & Muller is part of rpps.sync.time.MM.
"""
import numpy as np

def zx(early, mid, late):
    """Zero crossing (Gardner): `early` and `late` the instants of two symbols in a row, `mid` the sample half way between"""
    return np.real(mid*(np.conj(early) - np.conj(late)))

def el(before, at, after):
    """Early-late: `at` the symbol's instant, `before` and `after` half a symbol either side (the slope of |z|^2 there)"""
    return np.real(at*(np.conj(after) - np.conj(before)))

DETECTORS = {"zx": zx, "el": el}
