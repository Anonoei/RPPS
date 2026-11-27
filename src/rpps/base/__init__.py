"""The design: data, operators, categories, contracts and the streaming rules

RPPS stays signal agnostic: a new kind of signal should be new definitions plus existing operators.

## The model

    samples  ->  symbols  ->  channel bits  ->  data bits  ->  packets / messages

Channel bits are what crosses the channel (after FEC, interleaving, scrambling, sync markers); on receive they are LLRs
until the code transform decides them. Data bits are what the link layer carries.

- **Transforms** turn one representation into the next, as an invertible transmit/receive pair: wave, symbol, code, link.
- **Conditioning** (`signal`) changes samples without knowing what they carry.
- **Sync** estimates something unknown and corrects or aligns the data with it, at **any** representation: carrier and
  timing on samples, phase on symbols, codeword boundaries on channel bits, packet starts on data bits. A receiver uses it
  many times.

A **profile** (`rpps.profile`) puts these together for one signal, both directions. Data flows down the chain;
**control** flows back up it: a header decoded late can change earlier steps' parameters from a known position onward,
without losing lock (`rpps.base.control`, `rpps.profile.control`).

## The rule

A function belongs to the category named by **what it takes and what it gives**, not by what a standard calls it.

- Changes the representation: the transform for that step.
- Estimates an unknown and corrects or aligns with it: sync, at whatever representation.
- Changes samples without knowledge of the signal: signal.
- Inserting or removing known structure belongs to the transform where the structure lives: pilots and physical headers
  among symbols (symbol), sync markers among channel bits (code), headers and CRCs among data bits (link). Finding where
  it starts is sync.

Two implementations of one function share a signature, so they land in the same category.

## Categories

Receive direction; transmit runs the transforms in reverse, and sync has no transmit side.

| Category | Package | Takes → gives | Operator |
|---|---|---|---|
| Source | `source` | outside world ↔ any representation | `Source(n)`, `block >> sink` |
| Conditioning | `signal` | samples → samples | `Stage`, `+` |
| Sync | `sync` | any representation → the same, corrected or aligned; timing recovery takes samples to symbols | `Stage`, `+`, one-way |
| Wave | `wave` | samples ↔ symbols | `Stage` pairs |
| Symbol | `symbol` | symbols ↔ channel bits (LLRs on receive) | `Pipe`, `*` and `/` |
| Code | `code` | channel bits ↔ data bits | `Pipe`, `*` and `/` |
| Link | `link` | data bits ↔ packets | `Pipe`, `*` and `/` |
| Profile | `profile` | one whole signal, both directions | builds the above |
| Parallel | `parallel` | a profile's frame-parallel region | `profile.rx(parallel=...)` |

"Framing" is two things on opposite sides of the decoder: **sync markers** (CCSDS ASM, DVB-S sync bytes, DVB-S2 SOF,
802.15.4 SHR) live among symbols or channel bits, are found by sync and inserted by symbol or code; **link framing**
(HDLC, headers, lengths, CRCs) lives among data bits and belongs to link.

A DVB-S2 receiver in order: channelize, decimate (signal) → AGC, coarse frequency (sync) → matched filter (wave) → timing
(sync, gives symbols) → SOF, fine frequency and phase from pilots (sync) → PL descrambling, header and pilot removal,
demapping (symbol, gives LLRs) → deinterleaving, LDPC, BCH, BB descrambling (code) → BBHEADER, packets (link). DVB-S puts
sync between two code steps: Viterbi (code) → byte alignment and polarity from the sync bytes (sync) → Forney
deinterleaving, Reed-Solomon, energy dispersal (code).

## Contracts

`tests/unit/policy/test_contracts.py` checks these over every registered implementation.

- **source**: produces blocks at any representation and says which, and which sync was already applied (for example
  timed, carrier not removed). Sets the rates it knows (`Fs`, `cf`, symbol rate) and each block's absolute position.
  Sinks take any representation.
- **signal**: `SAMPLES` or `REAL` in, the same out. May change `Fs`, never needs `sps`, same in both directions. Includes
  `signal.channel`, the simulated medium tests use as the counterpart of sync.
- **sync**: any representation in, the same out, corrected or aligned. The one exception is **timing recovery**, the
  sampler (`SAMPLES` at `sps` > 1 → `SYMBOLS`); the matched filter before it is wave. Estimates (offset, timing, lock,
  SNR, detection, frame start, polarity) go to `meta.stream`, never into the data. No inverse.
- **wave**: transmit takes `SYMBOLS` and raises `sps` (shaping, FM/CPM, stagger, IFFT + CP); receive prepares samples for
  the sampler (matched filter, discriminator, rail realignment, CP removal + FFT) without choosing the instant. Comes in
  pairs (`rpps.wave`).
- **symbol**: encode takes channel bits, gives symbols. Decode gives `SYMBOLS` (differential, despreading, removing
  pilots and headers, descrambling) or `LLR` (demapping); the last symbol step always gives soft output.
- **code**: encode data bits → channel bits; decode channel bits (LLRs or hard) → data bits, LLRs when it can. FEC,
  interleaving, rate matching, puncturing, scrambling, line codes, sync markers; all invertible. Needs codeword
  boundaries known (finding them is sync).
- **link**: decode aligned data bits into packets (split at lengths or flags, check CRCs, strip headers and stuffing;
  status in `ctx`); encode builds them.

## Grey areas

Each settled by the rule.

| Function | Category | Why |
|---|---|---|
| Matched filter | wave | prepares samples; does not decide when to sample |
| Timing recovery (`OerderMeyr`, `MatTrack`, `Mat`, `Known`) | sync | estimates the instant; the one sync step that changes representation |
| Differential coding on symbols / on bits (NRZI) | symbol / code | symbols ↔ symbols / bits ↔ bits |
| Manchester as a pulse / as 0 → 01, 1 → 10 | wave / code | symbols ↔ samples / bits ↔ bits |
| Spreading at one sample per chip (`Spread`), `Despread` at `sps` with a known code phase | symbol | chips are symbols |
| Code-phase and Doppler search, preamble correlation, sync-marker and HDLC-flag search, phase ambiguity | sync | find a start or estimate an unknown |
| Attaching a sync marker | code | known structure among channel bits |
| Pilots, physical headers, padding, OFDM carrier allocation | symbol | known structure among symbols |
| Reading a physical header's fields (DVB-S2 PLS) | symbol | `symbol.Header` decodes its short code into fields |
| Equalizer, OFDM pilot equalization, squelch | sync | estimate the channel / presence |
| OFDM IFFT/FFT and CP; `FMDemod`, `AMDemod`, `SSBDemod` | wave | symbols ↔ samples, carrier ↔ baseband |
| De-emphasis, DC block; channel impairments | signal | samples → samples |
| CRC guarding a packet / inside FEC (LTE code-block CRC) | link / code | resources are shared across categories |

**Soft information only flows down the receive chain:** symbol gives LLRs, code takes and gives them, link takes hard
bits and their reliability, and sync on channel bits passes LLRs through. **Order matters inside sync:** a step sits
where what it corrects can still be separated (coarse frequency before the matched filter, fine phase after timing, a
staggered signal's carrier before its rails are realigned).

## Type tags

Every block carries a **form** (`dtype`: SAMPLES, REAL, SYMBOLS, LLR, BITS, BYTES, MSG; how values are stored) and a
**representation** (`Rep`: SAMPLES, SYMBOLS, CHANNEL, DATA, PACKETS; where in the chain). Forms cross representations:
LLRs exist before the decoder and after a soft-output one. Each operator declares `takes(encode)` and checks it before
working, so a link decoder refuses channel bits and a constellation refuses samples; `CATEGORY` says which category it is.

| Category | Accepts | Gives |
|---|---|---|
| signal | SAMPLES (form SAMPLES or REAL) | the same |
| sync | any | the same; timing gives SYMBOLS from SAMPLES |
| wave | SYMBOLS (tx), SAMPLES (rx) | SAMPLES |
| symbol | CHANNEL (encode), SYMBOLS (decode) | SYMBOLS; SYMBOLS or CHANNEL as LLR |
| code | DATA or CHANNEL (both ways) | CHANNEL; DATA as LLR or BITS |
| link | PACKETS (encode), DATA (decode) | DATA; PACKETS (BYTES or MSG) |

A code transform is often several steps, so code steps take channel and data bits alike; the enforced boundary is the
link's. `rp.code.Uncoded()` is the code step of a link with no FEC. `rp.Data(x)` on a complex array is SAMPLES,
`rp.Data(x, sps=1)` is SYMBOLS; `BitStage` takes channel or data bits and keeps their form.

## Data and operators

**Data** (`base/data.py`) wraps an array with its `Meta`: `Fs`, `sps`, centre frequency, group delay, stream state, and
`meta.pos`, the absolute position of its first item (`meta.time_of(k)` gives source time). Samples a source lost (a
radio's overrun, a dropped block) are counted in it, and a stage carries a gap in its input into its output positions,
so positions stay true to time. Nothing needs setting up or resetting.

| Operator | Means | On |
|---|---|---|
| `data * pipe`, `data / pipe` | encode / decode | `Pipe` (invertible transform) |
| `data + stage` | process | `Stage`, `BitStage` |
| `pipe * pipe`, `stage + stage` | a reusable `Pipeline` | either |
| `data >> sink` | write out | `Sink` |

A `Stage` keeps its settings on the object and everything that carries between blocks in `new_state(meta)`, held by the
stream; `step(samples, st, meta)` processes a block and sets `meta.op`. `SCOPE` is `STREAM` (state crosses frames) or
`FRAME`. `Stream` holds states, estimates (`stream.est[name]`), detections (`det`), control (`stream.control`) and load
statistics. `data + rp.Tee(a=..., b=...)` runs each branch in its own stream (FM stereo L+R and L-R).

## Streaming rules

Tested by cutting the same input differently (`tests/unit/base/`, per-stage tests):

1. **Any block length, including 0**; never drop or repeat items across blocks.
2. **State lives in the stream state**, so one stage can serve several streams.
3. **Whole blocks** through `rp.base.rpps.align(ctx, key, x, block, pad)`: the leftover waits, a flush pads (encode) or
   drops and reports (decode).
4. **Positions are absolute**; a stage that drops or inserts items anchors it.
5. **Feedforward estimators** follow the rules in `rpps.sync` (windows at absolute positions, tracked across windows).
6. **Soft information flows down**; sync on channel bits passes LLRs through.
"""
from .. import _config


from .data import Data, Digital, Mod, Analog
from .types import Category, Rep, dtype
from .meta import Meta, Stream, Rate

from .frequency import Frequency
from . import control
from .rpps import *
