from enum import Enum, auto
import numpy as np
import numpy.typing as nt

class pproc(Enum):
    """Previous process"""
    UNK = -1
    SERIAL = 0

    FILTER = 1
    SYNC = 2

    MOD = 4 # Modulated data
    MAP = 5 # Demodulated data

    CODING = 6
    SCRAM = 7

class dtype(Enum):
    """Data Type"""
    # Analog
    SAMPLES = (1, np.complex64)
    SYMBOLS = (11, np.complex64)

    # Mod
    MAPPED = (2, None)
    MAPPED_HARD = (21, None)
    MAPPED_SOFT = (22, None)

    # Digital
    BITS = (3, bool)
    BYTES = (4, np.uint8)

    MSG = (5, None)

    def __str__(self) -> str:
        return self.name

    @property
    def dtype(self) -> nt.DTypeLike:
        return self.value[1]
