# Modulation: two layers

A signal is built from explicit pieces, and a new kind of signal is a new config or a new piece, not a new special case.

| Layer | Pieces | Operator | What flows |
|---|---|---|---|
| **Symbol** (`rp.symbol`, `rp.spread`) | `Constellation`, `Differential`, `Spread`, `OFDM` | `*` encodes, `/` decodes | bits, symbols, chips (a `Pipe` each; decoding ends in LLRs) |
| **Waveform** (`rp.wave`, `rp.signal.sample`, `rp.signal.filters`) | `Zeros` / `Upsample`, pulse shaping, `Stagger`, `FM`, `FMDemod` | `+` | symbols and samples (a streaming `Stage` each; any block length) |

## Constellations
A constellation is a set of points and the ways of labelling them with bits. PSK, QAM, APSK and ASK are generators
(`rp.symbol.PSK(8)`, `rp.symbol.QAM(16)`, `rp.symbol.load("64QAM")`); anything else is a config file:

```json
{"base": "constellation",
 "points": [{"mag": 1, "pha": 45}, {"re": 0, "im": 2}, [-1.5, 0]],
 "maps": [{"map": [0, 1, 3, 2], "comment": "R000 Grey"}]}
```
`maps[i].map[point]` is the value (bits, first bit most significant) the point carries. Points are scaled to unit mean power unless
`"normalize": false`. The first map is also turned by every rotation the points allow, with and without spectral inversion
(`rp.symbol.load("QPSK", "R090 Inverted")`): the maps a receiver has to try when it does not know the carrier phase or the IQ sense.

## Differential coding
Explicit: `bits * rp.symbol.load("bpsk") * rp.symbol.Differential()` is DBPSK. The constellation's points are then the phase *changes*.
- DQPSK, 802.11b: `qpsk_axes` (points on the axes) + `Differential()`.
- pi/4-DQPSK: the ordinary 45 degree `QPSK` + `Differential()`.

## Frame structure
Known structure among the symbols of a frame (`rpps.symbol.frame`), restarted at each frame start a profile gives it:
- `Header`: known symbols then a short code carrying fields (`config/mod/header`, `rp.symbol.load("dvbs2_pl")`). Receiving it
  decodes the fields (every code word tried, soft) and publishes them for the profile; transmitting it builds them.
- `Pilots`: a block of pilots after every few slots (DVB-S2: 36 after every 16 slots of 90).
- `PhaseScramble`: each symbol turned by a multiple of 90 degrees from a binary sequence (DVB-S2 PL scrambling).

## Recipes
```python
# 802.11b 1 Mb/s: DBPSK, Barker-11, then 11 Mcps shaped as you like
chips = bits * rp.symbol.load("bpsk") * rp.symbol.Differential() * rp.symbol.Spread(rp.spread.load("barker11"))

# O-QPSK (half-sine pulses, Q half a symbol late; this is MSK when the symbols are unit QPSK)
wave = bits * rp.symbol.load("QPSK") + rp.signal.sample.up.Zeros(8) + rp.signal.filters.load("shaping", "halfsine") + rp.wave.Stagger()
# receive, coherent and in this order: remove the carrier phase, realign the rails, matched filter, sample
syms = wave + rp.wave.Stagger(rail="i") + rp.signal.filters.load("shaping", "halfsine") + rp.signal.sample.down.Decimate(8, offset=12)

# 802.15.4 2.4 GHz: 4 bits pick a 32-chip sequence, even chips on I, odd on Q
wave = bits * rp.symbol.Spread(rp.spread.load("ieee802154_2450"), pair=True) + rp.signal.sample.up.Zeros(4) \
       + rp.signal.filters.load("shaping", "halfsine") + rp.wave.Stagger()

# GFSK (Bluetooth LE: BT 0.5, h 0.5) and 4FSK: levels from an ASK constellation, a pulse, a frequency modulator
c = rp.symbol.ASK(2)
wave = bits * c + rp.signal.sample.up.Zeros(8) + rp.signal.filters.load("shaping", "gaussian/bt0p5") + rp.wave.FM(0.5, c.min_distance)
levels = wave + rp.wave.FMDemod(rp.wave.fm_gain(0.5, 8, c.min_distance))   # then integrate over each symbol and sample
```
Each is exercised in `tests/unit/test_recipes.py` and `tests/unit/test_wave.py`.

## Maths notes
### Quadrature
 - $x(t) = I\cos(2 \pi ft) + Q\sin(2 \pi ft)$
 - $A = \sqrt{ I^2+Q^2 }$, $\phi = \tan^{-1}(Q/I)$

### BPSK and QPSK error rates
 - BPSK: $P_b = Q(\sqrt{2E_b/N_0}) = \tfrac12 \text{erfc}(\sqrt{E_b/N_0})$
 - QPSK (Gray) has the same bit error rate per $E_b/N_0$; O-QPSK with matched filters too

### O-QPSK
Q is delayed by half a symbol, so I and Q never change together and the phase moves by at most 90 degrees at a time. With half-sine pulses
$p(t) = \sin(\pi t/T)$ on both rails the envelope is constant: it is MSK.
