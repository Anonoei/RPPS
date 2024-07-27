"""
Initialize RPPS
"""

__version__ = "0.0.3"
__author__ = "Anonoei <dev@anonoei.com>"


from pyboiler.logger import Logger, Level

Logger("RPPS", Level.TRACE)

# Import helpers/globals
from .helpers import bitarray, Stream
from .mod.meta import ModMeta
from .coding.meta import CodingMeta
from .meta import Meta


from . import coding
from . import mod
from . import viz

from .pipeline import Pipeline
