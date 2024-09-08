"""FEC/ECC encoders and decoders"""
# Import helpers/globals
from .. import base
from .. import dobject
from ..meta import Meta

# -----

from .coding import Coding, Block, Convolutional

from ._blk import *
from ._cnv import *

MAP = {
    "BLK": BLK_MAP,
    "CNV": CNV_MAP
}

from ._identify import name
