# Profiles: a known signal

A profile is one JSON file that describes a whole signal in both directions: its steps in receive order (filters, sync,
pulse, modulation, codes, framing) and the parameters a header may change along the way. When a signal has a profile,
use it instead of chaining operators by hand: the header-driven changes (lengths, rates, modulations) are wired up.

```
python -m rpps ls profile                 # the shipped profiles
python -m rpps show profile/ble/adv       # one, with its parents applied
python -m rpps steps                      # the step kinds and sync methods a profile can name
python -m rpps rx ble/adv capture.cf32 --fs 4e6 --param channel=38
python -m rpps tx ble/adv out.cf32 --rate 1e6 --random 20 --packet 20 --param channel=38 --field type=2
```

## Transmit and receive

```python
--8<-- "examples/profile_loopback.py"
```

## Writing one

A profile lives in `config/profile/<family>/<mode>.json` (or your own config directory: [Definitions](definitions.md)).
One list of steps, in receive order: a transform step (wave, symbol, code, link) runs as written on receive and
inverted, in reverse, on transmit; a sync step is skipped on transmit.

```json
{"base": "profile", "comment": "IEEE 802.15.4 2.4 GHz O-QPSK", "ref": "IEEE 802.15.4-2020 12",
 "params": {"sps": {"default": 4}},
 "defaults": {"carrier": {"method": "none"}},
 "steps": [
   {"sync": "carrier"},
   {"wave": "stagger", "delay": 0.5},
   {"wave": "shape", "pulse": "halfsine/halfsine", "sps": {"param": "sps"}},
   {"sync": "timing"},
   {"symbol": "spread", "sequence": "ieee802154_2450", "pair": true}
 ]}
```

builds `rx = carrier + Stagger("i") + halfsine matched + timing + Spread decode` and
`tx = Spread encode + Shape(4, halfsine) + Stagger("q")`. The carrier comes before `Stagger("i")`: a phase offset mixes
I into Q while Q is half a symbol late.

- A step that loads a definition names its kind and reference as the loaders take them:
  `{"code": "conv", "name": "k7_171_133/3_4"}`, `{"code": "interleave", "name": "forney/12_17"}`,
  `{"symbol": "constellation", "name": "psk/qpsk"}`.
- Sync steps name **what** is estimated (`carrier`, `timing`, `equalize`, `frame`) and get the first-class method; a
  loop is chosen in `defaults`: `{"timing": {"method": "gardner", "alpha": 0.3}}`.
- Any `Stage` can be a step by module path: `{"signal": "stage", "class": "mypkg.Gain", "args": {...}}`.
- `"controls"` lets a header's fields drive other steps (DVB-S2's MODCOD, 802.11b's rate): see
  [`rpps.profile.control`](../api/profile.md).
- Bursts, preambles, branches, resource grids and the parallel region are each a step or a key:
  [`rpps.profile`](../api/profile.md) lists them and where each is documented.

A new signal should need no new operator code. `examples/own_definitions.py` ([Definitions](definitions.md)) writes one
from scratch in a directory of your own.

## Decoded messages, and only some of them

Every frame a profile receives is also a record, `rp.profile.Message`: its kind, the fields it decoded, its payload,
`crc_ok`, and where it starts in the stream (`sample`, `time`, `freq`).

```python
# docs: skip
rx = rp.profile.load("adsb/mode_s").rx()
for block in src(65536):
    rx(block)
    for m in rx.take_messages():
        print(m.kind, m.crc_ok, m.fields, m.payload.hex() if m.payload else "")
```

`profile.kinds()` lists what a profile gives; `profile.rx(kinds={"pbch"})` (also `Decoder(..., kinds=)`,
`rpps rx --kinds pbch,pcfich`) hands out only those and builds only the steps they need.

## What a profile needs

`"requires"` says, as expressions over the params and `baud`, what a profile needs: `baud`, `Fs`, `bw`, `min_samples`,
`input` (`"audio"` for an FM channel's audio), `from_cf` (params the carrier gives). `rp.profile.info(name, baud=...)`
and `rp.profile.catalogue()` evaluate them, for a program building a menu. `rp.profile.Decoder` uses them to bring a
selected signal to the profile: shift and decimation, an FM demodulator for audio, a resampler to `Fs`, a rate measured
when the standard leaves it open.

## More than one core

A profile marks where its frame-parallel region starts; `p.rx(parallel=rp.parallel.Parallel(kind, workers=n))` runs it
in batches ([`rpps.parallel`](../api/parallel.md)).

```python
--8<-- "examples/parallel_rx.py"
```
