"""
RF Parameter Processor Suite

RPPS is a generic signal processor/generator library.


Github: https://github.com/Anonoei/RPPS

PyPI: https://pypi.org/project/rpps/
"""

__version__ = "0.3.0"
__author__ = "Anonoei <to+dev@an0.us>"

from . import logger
from . import errors
from . import _config
from ._config import add_config_path

# Import helpers/globals
from .base.types import pproc, dtype
from .base import data
from .base.data import Data
from .base import Pipeline, Pipe, Stage, Stream

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
from . import channel
from . import equalize
from . import demod
from .chain import Chain
from . import mod

from . import serial


def __getattr__(name):
    # Plotting pulls in matplotlib, so it's loaded on first use (rp.viz)
    if name == "viz":
        import importlib
        return importlib.import_module(".viz", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
