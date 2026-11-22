"""Synchronization stages: acquisition, AGC, PLLs, Costas, and phase (`phase`: carrier frequency and phase) and symbol-timing (`time`) recovery
"""
from .. import _config


from .costas import Costas
from .acquire import Acquire, Detection, Lock, detect, track_timing
from .time.mat import best_tau, matrix_mf, centres_in
from . import rate, quality
from .rate import find_baud, find_rate
from .acquire_fec import AcquireFEC
from .align import Align
from .puncture import Puncture

from . import phase
from . import time

from . import agc
from .agc import AGC, PeakAGC
from . import pll
from . import equalize
from .squelch import Squelch
from .code_acquire import acquire_code
from .pilots import PilotEqualizer, PilotPhase
from .frame import FrameSync, Delimiter
from .ofdm import CPSync
from .burst import Burst
from .tone import ToneLock
from .preamble import Preamble

from ..base.rpps import tag_stages as _tag_stages
from ..base.types import Category as _Category
_tag_stages("rpps.sync", _Category.SYNC)
