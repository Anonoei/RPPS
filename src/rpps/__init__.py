"""
RF Parameter Processor Suite

RPPS is a generic signal processor/generator library.


Github: https://github.com/Anonoei/RPPS

PyPI: https://pypi.org/project/rpps/
"""

__version__ = "0.2.0"
__author__ = "Anonoei <dev@anonoei.com>"

from pathlib import Path

from pyboiler.logger import Logger, Level
from pyboiler.config import config

# Definitions (config/) live next to src/ in a checkout; otherwise they're looked up from the running project's root
_checkout = Path(__file__).resolve().parents[2] / "config"
config().PATH_CONFIG = _checkout if _checkout.is_dir() else config().PATH_ROOT / "config" # type: ignore

Logger("RPPS", Level.TRACE)

# Import helpers/globals
from .base.types import pproc, dtype
from .base import data
from .base.data import Data
from .base.meta import Meta

from . import base
from . import math
from . import utils

## Import implementations
from . import filters
from . import dsp
from . import sample
from . import sync
from . import analysis

from . import scram
from . import coding
from . import encap
from . import chain
from .chain import Chain
from . import mod

from . import serial

from . import viz
