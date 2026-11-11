"""Modulation: the symbol layer (bits <-> symbols, `*` and `/`); the waveform layer is rpps.wave"""
# Import helpers/globals
from .. import base
from .. import _config

# -----
from . import shapes
from .mapping import Mapping
from .constellation import Constellation, PSK, QAM, APSK, ASK
from .differential import Differential
from .spread import Spread
from .despread import Despread, correlate
from .ofdm import Carriers
from .frame import Header, Pilots, PhaseScramble, pilot_layout, rotating
from .serial import ls, load, init, generate

from .modulation import Modulation
