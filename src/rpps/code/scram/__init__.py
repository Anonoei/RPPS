"""Scramblers, a step of the code transform: additive (a sequence XOR'd in) and self-synchronizing (feedthrough)"""

# Import helpers/globals
from ... import base
from ... import _config

# -----

from .serial import ls, load

from . import lfsr
from .scrambler import Scram
from .sequence import Sequence
