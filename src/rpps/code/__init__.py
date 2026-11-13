"""Code transform: channel bits <-> data bits (README.dev.md)

FEC encoders and decoders, interleaving, rate matching, scrambling (`code.scram`), and `Uncoded` for links with no FEC.
Each step takes either channel or data bits, because a code transform is often several steps.
"""
# Import helpers/globals
from .. import base
from .. import _config

# -----
from . import types
from . import matrix
from .serial import ls, names, load, init, find, generate

from .coding import Coding, Block, Convolutional, Interleaver
from .uncoded import Uncoded

from .factory import Hamming, Repetition, RS, Conv, Cyclic, Golay
from .alist import import_alist, read_alist
from . import scram
