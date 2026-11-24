"""Profiles: a whole signal as one config file, in receive order, both directions

    p = rp.profile.load("ieee802154/2450", sps=4)
    wave = p.tx()(rp.Data(payload))             # transmit: the transforms inverted, in reverse order
    bits = p.rx()(wave)                          # receive: every step as written
    bits = p.rx(start=rp.Rep.SYMBOLS)(symbols)   # a source that gives timed symbols: the steps before are skipped

A profile is `config/profile/<family>/<mode>.json`: operators, order, parameters. Writing one needs no code; the guide
"Profiles" walks through one, and `python -m rpps steps` lists every step kind and sync method.

## One list of steps, in receive order

A transform step names a pair, runs as written on receive and inverted, in reverse, on transmit; a sync step is skipped on
transmit. Profiles take `params` like any definition; `defaults` holds sync methods and loop settings. A step that loads a
definition names its kind and reference as the loaders take them: `{"code": "conv", "name": "k7_171_133/3_4"}`,
`{"symbol": "constellation", "name": "psk/qpsk"}`. The kinds are in `rpps.profile.steps`; how a wave step pairs on
receive and transmit is in `rpps.wave`.

## What else a profile can say

| Feature | Says | Where |
|---|---|---|
| Variants | a param's `"choices"` (each a bundle of fields: `{"param": "modulation.shape"}`), a step's `"when"` | `rpps._config.resolve`, guide "Definitions" |
| Header-driven control | `"controls"`: a header's fields drive other steps' parameters from a position on, without losing lock | `rpps.profile.control`, `rpps.base.control`, `rpps.profile.runner` |
| Bursts | `{"sync": "burst"}`: the steps after it are built anew for every burst; transmitting places bursts on one stream (`gap_s`, `at=`, `ramp_s`, `freq=`, `hop`) | `rpps.profile.burst`, `rpps.sync.Burst`, `rpps.sync.BurstTx` |
| Preambles | `{"sync": "preamble"}` after a burst step: by repetition or a waveform the profile makes from a `sequence` word; `"attach": true` sends it on transmit | `rpps.sync.Preamble` |
| Branches | a last `{"branch": ...}` step fans the chain out; a header in one branch drives the branches after it, `frame` makes them read whole frames, `take` only parts of each (a code split around a sync word), `select` only the frames a trunk step's fields accept | `rpps.profile.branch` |
| Resource grids | `grid` steps take an OFDM system's samples to frames of its grid; its channels are branches (`lte/downlink`, `nr/downlink`) | `rpps.grid` |
| Parallel region | `"parallel": "<step id>"`: the per-frame rest of the chain on threads or processes | `rpps.parallel`, `rpps.profile.parallel` |
| Messages | every frame is also a `Message` (kind, fields, payload, `crc_ok`, where it starts), from `runner.take_messages()` | `rpps.profile.message` |
| Only some kinds | `profile.kinds()`, `profile.rx(kinds={"pbch"})`: hands out only those and builds only the steps they need (`profile.needed`) | `rpps.profile.profile` |
| Requirements | `"requires"`: `baud`, `Fs`, `bw`, `min_samples`, `input`, `from_cf` as expressions, evaluated when asked | `rpps.profile.catalogue` (`fits`, `candidates`: what a measured signal can be), `rpps.profile.Decoder` |

**Sources that start later in the chain:** the runner skips every receive step before what the source gives
(`Source.REP`, `APPLIED`): timed symbols with the carrier on skip signal, wave and timing; a BBFRAME stream skips
everything through code.

**Messages** come from a step that knows where a frame starts and whether it checked out
(`rp.base.control.emit(meta, op, kind, fields, bits, ok, at)`: link headers, failed frames too, the HDLC frame step, the
DVB-S2 PL header, the LTE and NR channels). A run of fixed-size frames with no step of their own (TS packets after DVB-S's
RS) is cut by the runner from a step's `"message"`: `{"kind": "packet", "bits": 1504, "check": true, "fields": [...]}`.
The runner places each message through the maps of the steps before it and hands it out once; burst and branch runners
place their children's in the outer stream.

**Only some kinds:** a branch none of whose steps is needed is not built, and a chain ends at its last needed step, so
the runner's output is that step's (DVB-S2's `pls` alone: no descrambling, LDPC or BCH). LTE's PDSCH needs the PDCCH,
which needs the PCFICH, which needs the PBCH. A kind the profile doesn't give is a ConfigError.
"""
from .profile import Profile, load, ls
from .message import Message
from .catalogue import ProfileInfo, info, catalogue, variants, fits, candidates
from .decoder import Decoder
from .runner import Runner
from .parallel import Region
from .steps import Step
from .control import table, Table
