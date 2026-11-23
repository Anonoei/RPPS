r"""Modulation: the symbol layer (channel bits <-> symbols, `*` and `/`); the waveform layer is rpps.wave

A signal is built from explicit pieces, and a new kind of signal is a new config or a new piece, not a new special case.

| Layer | Pieces | Operator | What flows |
|---|---|---|---|
| **Symbol** (`rp.symbol`; codes from `rp.sequence`) | `Constellation`, `Differential`, `Spread`, `Carriers` | `*` encodes, `/` decodes | bits, symbols, chips (a `Pipe` each; decoding ends in LLRs) |
| **Waveform** (`rp.wave`; pulses from `rp.signal.filters.load("pulse/...")`) | `Shape` / `Matched`, `Stagger`, `FM`, `FMDemod` | `+` | symbols and samples (a streaming `Stage` each; any block length) |

## Constellations

A constellation is a set of points and the ways of labelling them with bits. PSK, QAM, APSK and ASK are generators
(`rp.symbol.PSK(8)`, `rp.symbol.QAM(16)`, `rp.symbol.generate("64QAM")`); anything else is a config file:

    {"base": "constellation",
     "points": [{"mag": 1, "pha": 45}, {"re": 0, "im": 2}, [-1.5, 0]],
     "maps": [{"map": [0, 1, 3, 2], "comment": "R000 Grey"}]}

`maps[i].map[point]` is the value (bits, first bit most significant) the point carries. Points are scaled to unit mean
power unless `"normalize": false`. The first map is also turned by every rotation the points allow, with and without
spectral inversion (`rp.symbol.load("psk/qpsk", "R090 inverted")`): the maps a receiver has to try when it does not know
the carrier phase or the IQ sense. Shapes and definitions are separate: DVB-S2's 16APSK is
`rp.symbol.load("apsk/dvbs2_16apsk_3_4")`, not `generate("16APSK")`.

## Differential coding

Explicit: `bits * rp.symbol.load("psk/bpsk") * rp.symbol.Differential()` is DBPSK; the constellation's points are then the
phase *changes*. 802.11b DQPSK is `psk/qpsk_axes` + `Differential()`; pi/4-DQPSK the ordinary 45 degree `psk/qpsk` +
`Differential()`. Bit-level differential coding (NRZ-M, NRZ-I) is `rp.code.Line`.

## Frame structure

Known structure among the symbols of a frame (`rpps.symbol.frame`), restarted at each frame start a profile gives it:

- `Header`: known symbols then a short code carrying fields (`config/header/physical`,
  `rp.symbol.load_header("dvbs2_pl")`). Receiving it decodes the fields (every code word tried, soft) and publishes them
  for the profile; transmitting it builds them.
- `Pilots`: a block of pilots after every few slots (DVB-S2: 36 after every 16 slots of 90).
- `PhaseScramble`: each symbol turned by a multiple of 90 degrees from a binary sequence (DVB-S2 PL scrambling).

## Recipes

Each is exercised in `tests/unit/test_recipes.py` and `tests/unit/test_wave.py`. Shaping needs the symbols' rate, so the
bits say theirs: `bits = rp.Data(b, Fs=1e6, sps=1)`.

    # 802.11b 1 Mb/s: DBPSK, Barker-11, then 11 Mcps shaped as you like
    chips = bits * rp.symbol.load("psk/bpsk") * rp.symbol.Differential() * rp.symbol.Spread(rp.sequence.load("barker11"))

    # O-QPSK (half-sine pulses, Q half a symbol late; MSK when the symbols are unit QPSK)
    halfsine = rp.signal.filters.load("pulse/halfsine/halfsine")
    wave = bits * rp.symbol.load("psk/qpsk") + rp.wave.Shape(8, halfsine) + rp.wave.Stagger()

    # 802.15.4 2.4 GHz: 4 bits pick a 32-chip sequence, even chips on I, odd on Q
    wave = bits * rp.symbol.Spread(rp.sequence.load("ieee802154_2450"), pair=True) + rp.wave.Shape(4, halfsine) + rp.wave.Stagger()

    # GFSK (Bluetooth LE: BT 0.5, h 0.5): levels from an ASK constellation, a pulse, a frequency modulator
    c = rp.symbol.ASK(2)
    wave = bits * c + rp.wave.Shape(8, rp.signal.filters.load("pulse/gaussian/bt0p5")) + rp.wave.FM(0.5, c.min_distance)
    levels = wave + rp.wave.FMDemod(rp.wave.fm_gain(0.5, 8, c.min_distance))   # then integrate over each symbol and sample

## Notes

- BPSK: $P_b = Q(\sqrt{2E_b/N_0}) = \tfrac12 \text{erfc}(\sqrt{E_b/N_0})$; QPSK (Gray) has the same bit error rate per
  $E_b/N_0$, O-QPSK with matched filters too.
- O-QPSK delays Q by half a symbol, so I and Q never change together and the phase moves by at most 90 degrees at a time.
  With half-sine pulses $p(t) = \sin(\pi t/T)$ on both rails the envelope is constant: it is MSK.
"""
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
from .serial import ls, load, load_header, init, generate, load_or_generate

from .modulation import Modulation
