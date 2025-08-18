"""(De-)Scrambler implementations"""

# Import helpers/globals
from .. import base
from .. import _config

# -----

from .serial import identify, load

from . import lfsr
from .scrambler import Scram
