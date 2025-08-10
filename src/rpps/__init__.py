"""
RF Parameter Processor Suite

RPPS is a generic signal processor/generator library.


Github: https://github.com/Anonoei/RPPS

PyPI: https://pypi.org/project/rpps/
"""

__version__ = "0.2.0"
__author__ = "Anonoei <dev@anonoei.com>"

from pyboiler.logger import Logger, Level
from pyboiler.config import config

config().PATH_CONFIG = config().PATH_ROOT / "config" # type: ignore

Logger("RPPS", Level.TRACE)

# Import helpers/globals
from . import base
from . import utils
from . import dobject

from .meta import Meta

## Import implementations
from . import filters
from . import sample
from . import sync

from . import scram
from . import coding
from . import mod

from . import serial

from . import viz
