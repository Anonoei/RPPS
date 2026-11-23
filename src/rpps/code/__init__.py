"""Code transform: channel bits <-> data bits

FEC encoders and decoders, interleaving, rate matching, scrambling (`code.scram`), line codes (`Line`: NRZI, Manchester,
inversion), sync markers among channel bits (`Marker`), and `Uncoded` for links with no FEC.
Each step takes either channel or data bits, because a code transform is often several steps.
"""
# Import helpers/globals
from .. import base
from .. import _config

# -----
from . import types
from .serial import ls, names, load, init, find, locate

from .coding import Coding, Block, Convolutional, Interleaver, puncture_period, mismatch, cached
from .uncoded import Uncoded
from .line import Line
from .marker import Marker

from .factory import Hamming, Repetition, RS, Conv, Cyclic, Golay
from .alist import import_alist, read_alist
from . import scram
