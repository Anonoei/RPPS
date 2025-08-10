"""FEC/ECC encoders and decoders"""
# Import helpers/globals
from .. import base
from .. import dobject

# -----
from .serial import identify, load, generate

from .coding import Coding, Block, Convolutional
