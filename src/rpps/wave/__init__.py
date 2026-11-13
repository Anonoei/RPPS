"""Waveform layer: streaming stages between symbols and samples (`data + stage`)

    Stagger   offset one rail by part of a symbol (O-QPSK)
    FM        a shaped level signal to a phase-continuous frequency-modulated one (FSK, GFSK, GMSK, C4FM)
    FMDemod   its quadrature discriminator

Pulse shapes are in rp.signal.filters (`halfsine`, `gaussian`, `rrc`, `rect`), the rate change is rp.signal.sample.up.Zeros:

    qpsk = rp.symbol.load("QPSK")
    wave = bits * qpsk + rp.signal.sample.up.Zeros(8) + rp.signal.filters.load("pulse", "halfsine") + rp.wave.Stagger()   # O-QPSK
"""
from .stagger import Stagger
from .fm import FM, FMDemod, fm_gain
from .analog import AMDemod, SSBDemod
from .ofdm import OFDMTx, OFDMRx

from ..base.rpps import tag_stages as _tag_stages
from ..base.types import Category as _Category
_tag_stages("rpps.wave", _Category.WAVE)
