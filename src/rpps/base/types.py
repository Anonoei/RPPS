from enum import Enum, auto
import numpy as np

class dtype(Enum):
    samples = np.complex64
    symbols = np.complex64
    mapped = None
    mapped_hard = None
    mapped_soft = None
    bytes = np.uint8
    bits = bool

    def __str__(self):
        return self.name

class ptype(Enum):
    """Data Types"""
    UNKNOWN = -1
    SAMPLES = 0
    SYMBOLS = 1
    MAPPED = 2
    MAPPED_HARD = 21
    MAPPED_SOFT = 22
    CODED = 3
    FRAME = 4
    MSG = 9

    def __str__(self):
        return self.name
