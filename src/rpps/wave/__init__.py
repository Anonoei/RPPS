"""Waveform layer: streaming stages between symbols and samples (`data + stage`)

    Shape     symbols to samples, each symbol carrying a pulse (rp.signal.filters.load("pulse/..."): the definitions)
    Matched   the pulse's matched filter
    Stagger   offset one rail by part of a symbol (O-QPSK)
    FM        a shaped level signal to a phase-continuous frequency-modulated one (FSK, GFSK, GMSK, C4FM)
    FMDemod   its quadrature discriminator
    demod     FM, AM or SSB audio from one block, no state, with FM deviation, AM depth and SINAD

The pulses are definitions (config/pulse: `halfsine`, `gaussian`, `rrc`, `rect`), loaded by rp.signal.filters.load:

    qpsk = rp.symbol.load("psk/qpsk")
    wave = bits * qpsk + rp.wave.Shape(8, rp.signal.filters.load("pulse/halfsine/halfsine")) + rp.wave.Stagger()   # O-QPSK

## Wave pairs

A profile's wave step runs as written on receive and inverted on transmit:

| Step | Receive (as written) | Transmit (inverted) |
|---|---|---|
| `shape`, symbol pulse (rrc, rc, halfsine, rect) | `wave.Matched(pulse)`, unit gain at the instants | `wave.Shape(sps, pulse)`, polyphase |
| `shape`, frequency pulse (gaussian, FSK rect) | none (the `fm` discriminator) | `wave.Shape(sps, pulse)` |
| `stagger` | `Stagger("i", delay)` | `Stagger("q", delay)` |
| `fm` | `FMDemod` + integrate over the symbol | `FM(h, step)` |
| `ofdm` | remove CP + FFT | IFFT + CP |

Whether a pulse is a symbol or a frequency pulse is the pulse definition's `kind`, not the profile's.
"""
from .shape import Shape, Matched
from .stagger import Stagger
from .fm import FM, FMDemod, fm_gain, discriminate
from .analog import AMDemod, SSBDemod, demod, Measurements
from .ofdm import OFDMTx, OFDMRx

from ..base.rpps import tag_stages as _tag_stages
from ..base.types import Category as _Category
_tag_stages("rpps.wave", _Category.WAVE)
