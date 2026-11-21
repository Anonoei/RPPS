"""Waveform layer: streaming stages between symbols and samples (`data + stage`)

    Shape     symbols to samples, each symbol carrying a pulse (rp.signal.filters.load("pulse/..."): the definitions)
    Matched   the pulse's matched filter
    Stagger   offset one rail by part of a symbol (O-QPSK)
    FM        a shaped level signal to a phase-continuous frequency-modulated one (FSK, GFSK, GMSK, C4FM)
    FMDemod   its quadrature discriminator

The pulses are definitions (config/pulse: `halfsine`, `gaussian`, `rrc`, `rect`), loaded by rp.signal.filters.load:

    qpsk = rp.symbol.load("psk/qpsk")
    wave = bits * qpsk + rp.wave.Shape(8, rp.signal.filters.load("pulse/halfsine")) + rp.wave.Stagger()   # O-QPSK
"""
from .shape import Shape, Matched
from .stagger import Stagger
from .fm import FM, FMDemod, fm_gain, discriminate
from .analog import AMDemod, SSBDemod
from .ofdm import OFDMTx, OFDMRx

from ..base.rpps import tag_stages as _tag_stages
from ..base.types import Category as _Category
_tag_stages("rpps.wave", _Category.WAVE)
