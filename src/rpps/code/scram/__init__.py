"""Scramblers, a step of the code transform: additive (a sequence of rpps.sequence XOR'd in) and self-synchronizing (feedthrough)"""

# Import helpers/globals
from ... import base
from ... import _config

# -----

from .serial import ls, load

from .scrambler import Scram, Additive, Feedthrough
