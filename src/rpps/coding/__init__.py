"""FEC/ECC encoders and decoders"""
# Import helpers/globals
from .. import base
from .. import _config

# -----
from . import err
from . import matrix
from .serial import identify, load, generate

from .coding import Coding, Block, Convolutional
