"""FEC/ECC encoders and decoders"""
# Import helpers/globals
from .. import base
from .. import _config

# -----
from . import err
from . import types
from . import matrix
from .serial import ls, load, generate

from .coding import Coding, Block, Convolutional
