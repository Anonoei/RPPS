"""Synchronization stages: acquisition, AGC, PLLs, Costas, and frequency (`freq`), phase (`phase`) and symbol-timing (`time`) recovery
"""
from .. import _config

from .sync import Sync

from .costas import Costas
from .acquire import Acquire

from . import freq
from . import phase
from . import time

from . import agc
from .agc import AGC
from . import pll
