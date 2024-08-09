"""
Initialize RPPS
"""

__version__ = "0.0.5"
__author__ = "Anonoei <dev@anonoei.com>"


from pyboiler.logger import Logger, Level

Logger("RPPS", Level.TRACE)

# Import helpers/globals
from .helpers import bitarray, Stream, progress
from .mod.meta import ModMeta
from .coding.meta import CodingMeta
from .meta import Meta

from .file import file


from . import coding
from . import mod
from . import viz
from . import freq
from . import process

from .pipeline import Pipeline
