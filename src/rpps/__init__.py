"""
RF Parameter Processor Suite

RPPS is a generic signal processor/generator library.


Github: https://github.com/Anonoei/RPPS

PyPI: https://pypi.org/project/rpps/
"""

__version__ = "0.1.0"
__author__ = "Anonoei <dev@anonoei.com>"


from pyboiler.logger import Logger, Level
from pyboiler.config import config

config().PATH_CONFIG = config().PATH_ROOT / "config" # type: ignore

Logger("RPPS", Level.TRACE)

# Import helpers/globals
from . import base
from . import helpers

from . import dobject

## Import implementations

from .file import file

from . import scram
from . import coding
from . import mod
from . import viz
from . import process
