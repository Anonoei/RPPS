"""Type tags every block of data carries (see README.dev.md, Type tags)

    Category  the kind of operation that made the data last
    Rep       where in the chain the data is: samples -> symbols -> channel bits -> data bits -> packets
    dtype     how the values are stored (the form): complex samples, LLRs, bits, bytes...

Rep and dtype are separate because forms cross representations: LLRs and hard bits exist both before the decoder
(channel bits) and after it (data bits).
"""
from enum import Enum
import numpy as np
import numpy.typing as nt

class Category(Enum):
    """The kind of operation that produced the data"""
    UNK = -1
    SOURCE = 0
    SIGNAL = 1
    SYNC = 2
    WAVE = 3
    SYMBOL = 4
    CODE = 5
    LINK = 6

    def __str__(self) -> str:
        return self.name

class Rep(Enum):
    """Where in the chain the data is"""
    SAMPLES = 0  # samples at sps >= 1 (complex IQ, or real: demodulated FM, audio)
    SYMBOLS = 1  # one value per symbol (or chip)
    CHANNEL = 2  # bits as they go over the channel: before decoding (LLRs on receive)
    DATA = 3     # bits the link layer carries: decoded
    PACKETS = 4  # packets and messages

    def __str__(self) -> str:
        return self.name

class dtype(Enum):
    """Data Type"""
    # Analog
    SAMPLES = (1, np.complex64)
    SYMBOLS = (11, np.complex64)
    REAL = (12, np.float32) # Real samples (demodulated FM, audio...)

    # Mod
    MAPPED = (2, None)
    MAPPED_HARD = (21, bool) # Hard decided bits
    LLR = (22, np.float32)   # Soft bits: log(P(0)/P(1)) per bit, positive means 0

    # Digital
    BITS = (3, bool)
    BYTES = (4, np.uint8)

    MSG = (5, None)

    def __str__(self) -> str:
        return self.name

    @property
    def dtype(self) -> nt.DTypeLike:
        return self.value[1]

# The representation a form has when nothing says otherwise: bits from the user are data bits, demodulated bits are
# channel bits. A code transform says which side of the decoder its bits are on (see `_Data.derive`)
REP_OF = {
    dtype.SAMPLES: Rep.SAMPLES, dtype.REAL: Rep.SAMPLES, dtype.SYMBOLS: Rep.SYMBOLS,
    dtype.MAPPED: Rep.CHANNEL, dtype.MAPPED_HARD: Rep.CHANNEL, dtype.LLR: Rep.CHANNEL,
    dtype.BITS: Rep.DATA, dtype.BYTES: Rep.DATA, dtype.MSG: Rep.PACKETS,
}
