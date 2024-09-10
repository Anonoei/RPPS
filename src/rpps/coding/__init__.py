"""FEC/ECC encoders and decoders"""
# Import helpers/globals
from .. import base
from .. import dobject
from ..meta import Meta

# -----

from . import conv

from .coding import Coding, Block, Convolutional

from .blk import *
from .cnv import *

MAP = {
    "BLK": BLK_MAP,
    "CNV": CNV_MAP
}

from ._identify import name
