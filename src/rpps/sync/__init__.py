"""Synchronization stages: acquisition, AGC, PLLs, Costas, and frequency (`freq`), phase (`phase`) and symbol-timing (`time`) recovery
"""
from .. import _config


from .costas import Costas
from .acquire import Acquire
from .align import Align
from .puncture import Puncture

from . import freq
from . import phase
from . import time

from . import agc
from .agc import AGC
from . import pll
from . import equalize
from .squelch import Squelch
from .code_acquire import acquire_code
from .pilots import PilotEqualizer, PilotPhase
from .frame import FrameSync, Delimiter

from ..base.rpps import tag_stages as _tag_stages
from ..base.types import Category as _Category
_tag_stages("rpps.sync", _Category.SYNC)
