# RPPS development guide

How RPPS is designed and built, and how to add to it. [README.md](README.md) is for users; [config/README.md](config/README.md)
covers definition files.

- [Setup and workflow](#setup-and-workflow) · [Repository layout](#repository-layout)
- [Design](#design): [the model](#the-model), [the rule](#the-rule), [categories](#categories), [contracts](#contracts),
  [grey areas](#grey-areas), [type tags](#type-tags), [where things are](#where-things-are)
- [Sync methods](#sync-methods) · [Profiles](#profiles) · [Parallelism](#parallelism)
- [Core abstractions and streaming rules](#core-abstractions-and-streaming-rules) · [Definitions](#definitions-config) ·
  [Effort](#effort) · [Native kernels](#native-kernels)
- [Testing](#testing) · [Adding things](#adding-things) · [CI and releases](#ci-and-releases) · [Conventions](#conventions)
- [Not yet built](#not-yet-built) · [Blind analysis](#blind-analysis-rppsanalysis)

## Setup and workflow

```
git clone https://github.com/Anonoei/RPPS && cd RPPS
python builder.py -s        # pip install -e ".[dev]"
python builder.py -c        # build the C++ kernels in place (C++17 compiler)
python builder.py -t        # pytest
```

`builder.py -h` lists the rest: `-d` docs, `-b` sdist and wheel (then `twine check`), `-l` local install, `-v M|m|p` version
bump and `-u` upload (both dry runs without `-r`; upload goes to TestPyPI without it).

`pyproject.toml` is the only place that names a dependency, CI included: `[build-system]` (setuptools, numpy, pybind11),
`dependencies` (numpy, pysdrlib), extras `viz` (matplotlib), `test` (pytest, pytest-xdist), `oracle` (scipy), `docs` (pdoc3),
`dev` (all of them plus twine, build, bumpver). It also holds `[tool.cibuildwheel]`, `[tool.bumpver]` and the pytest markers.

Branch from `dev`, merge back into it; `main` is released. Scratch scripts are not committed; worked examples that reuse
RPPS and may be upstreamed go in `projects/`.

## Repository layout

```
src/rpps/            the package: one sub-package per category (Where things are)
csrc/rpps_native/    C++ kernels, built as rpps._native (optional)
config/              JSON definitions, copied into the wheel as rpps/config
tests/unit/          the test suite
tests/*.py           benchmarks (bench_stages.py, parallel_bench.py, tpc_ber.py)
examples/            one script per recommended way of doing a thing, each checking itself (examples/README.md)
projects/            worked examples on real signals (FM, ADS-B, ATSC, LTE, P25 ...), each with a README
data/                captures (git-ignored)
builder.py, setup.py local build helper; setup.py builds the native module and copies config/
```

`python -m rpps` has `ls`, `show <ref>`, `steps`, `rx`, `tx`, `survey`, `profile`, `solve` and `fmt`. In code,
`rp.load("<resource>/<name>", **params)` loads any definition.

## Design

RPPS stays signal agnostic: a new kind of signal should be new definitions plus existing operators.

### The model

```
samples  ->  symbols  ->  channel bits  ->  data bits  ->  packets / messages
```

Channel bits are what crosses the channel (after FEC, interleaving, scrambling, sync markers); on receive they are LLRs until
the code transform decides them. Data bits are what the link layer carries.

- **Transforms** turn one representation into the next, as an invertible transmit/receive pair: wave, symbol, code, link.
- **Conditioning** (`signal`) changes samples without knowing what they carry.
- **Sync** estimates something unknown and corrects or aligns the data with it, at **any** representation: carrier and timing on
  samples, phase on symbols, codeword boundaries on channel bits, packet starts on data bits. A receiver uses it many times.

A **profile** puts these together for one signal, both directions. Data flows down the chain; **control** flows back up it: a
header decoded late can change earlier steps' parameters from a known position onward, without losing lock
([Header-driven control](#header-driven-control)).

### The rule

A function belongs to the category named by **what it takes and what it gives**, not by what a standard calls it.

- Changes the representation: the transform for that step.
- Estimates an unknown and corrects or aligns with it: sync, at whatever representation.
- Changes samples without knowledge of the signal: signal.
- Inserting or removing known structure belongs to the transform where the structure lives: pilots and physical headers among
  symbols (symbol), sync markers among channel bits (code), headers and CRCs among data bits (link). Finding where it starts is sync.

Two implementations of one function share a signature, so they land in the same category.

### Categories

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

"Framing" is two things on opposite sides of the decoder: **sync markers** (CCSDS ASM, DVB-S sync bytes, DVB-S2 SOF, 802.15.4
SHR) live among symbols or channel bits, are found by sync and inserted by symbol or code; **link framing** (HDLC, headers,
lengths, CRCs) lives among data bits and belongs to link.

A DVB-S2 receiver in order: channelize, decimate (signal) → AGC, coarse frequency (sync) → matched filter (wave) → timing
(sync, gives symbols) → SOF, fine frequency and phase from pilots (sync) → PL descrambling, header and pilot removal,
demapping (symbol, gives LLRs) → deinterleaving, LDPC, BCH, BB descrambling (code) → BBHEADER, packets (link). DVB-S puts
sync between two code steps: Viterbi (code) → byte alignment and polarity from the sync bytes (sync) → Forney
deinterleaving, Reed-Solomon, energy dispersal (code).

### Contracts

- **source**: produces blocks at any representation and says which, and which sync was already applied (for example timed,
  carrier not removed). Sets the rates it knows (`Fs`, `cf`, symbol rate) and each block's absolute position. Files,
  sockets, radios through pysdrlib. Sinks take any representation.
- **signal**: `SAMPLES` or `REAL` in, the same out. May change `Fs`, never needs `sps`, same in both directions. Includes
  `signal.channel`, the simulated medium tests use as the counterpart of sync.
- **sync**: any representation in, the same out, corrected or aligned. The one exception is **timing recovery**, the sampler
  (`SAMPLES` at `sps` > 1 → `SYMBOLS`); the matched filter before it is wave. Estimates (offset, timing, lock, SNR, detection,
  frame start, polarity) go to `meta.stream`, never into the data. No inverse.
- **wave**: transmit takes `SYMBOLS` and raises `sps` (shaping, FM/CPM, stagger, IFFT + CP); receive prepares samples for the
  sampler (matched filter, discriminator, rail realignment, CP removal + FFT) without choosing the instant. Comes in
  [pairs](#wave-pairs).
- **symbol**: encode takes channel bits, gives symbols. Decode gives `SYMBOLS` (differential, despreading, removing pilots and
  headers, descrambling) or `LLR` (demapping); the last symbol step always gives soft output.
- **code**: encode data bits → channel bits; decode channel bits (LLRs or hard) → data bits, LLRs when it can. FEC,
  interleaving, rate matching, puncturing, scrambling, line codes, sync markers; all invertible. Needs codeword boundaries
  known (finding them is sync).
- **link**: decode aligned data bits into packets (split at lengths or flags, check CRCs, strip headers and stuffing; status in
  `ctx`); encode builds them.

### Grey areas

Each settled by the rule.

| Function | Category | Why |
|---|---|---|
| Matched filter | wave | prepares samples; does not decide when to sample |
| Timing recovery (`MM`, `Mat`, `TLL`, `OerderMeyr`) | sync | estimates the instant; the one sync step that changes representation |
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

**Soft information only flows down the receive chain:** symbol gives LLRs, code takes and gives them, link takes hard bits and
their reliability, and sync on channel bits passes LLRs through. **Order matters inside sync:** a step sits where what it
corrects can still be separated (coarse frequency before the matched filter, fine phase after timing, a staggered signal's
carrier before its rails are realigned).

### Type tags

Every block carries a **form** (`dtype`: SAMPLES, REAL, SYMBOLS, LLR, BITS, BYTES, MSG; how values are stored) and a
**representation** (`Rep`: SAMPLES, SYMBOLS, CHANNEL, DATA, PACKETS; where in the chain). Forms cross representations: LLRs
exist before the decoder and after a soft-output one. Each operator declares `takes(encode)` and checks it before working, so a
link decoder refuses channel bits and a constellation refuses samples; `CATEGORY` says which category it is.

| Category | Accepts | Gives |
|---|---|---|
| signal | SAMPLES (form SAMPLES or REAL) | the same |
| sync | any | the same; timing gives SYMBOLS from SAMPLES |
| wave | SYMBOLS (tx), SAMPLES (rx) | SAMPLES |
| symbol | CHANNEL (encode), SYMBOLS (decode) | SYMBOLS; SYMBOLS or CHANNEL as LLR |
| code | DATA or CHANNEL (both ways) | CHANNEL; DATA as LLR or BITS |
| link | PACKETS (encode), DATA (decode) | DATA; PACKETS (BYTES or MSG) |

A code transform is often several steps, so code steps take channel and data bits alike; the enforced boundary is the link's.
`rp.code.Uncoded()` is the code step of a link with no FEC. `rp.Data(x)` on a complex array is SAMPLES, `rp.Data(x, sps=1)` is
SYMBOLS; `BitStage` takes channel or data bits and keeps their form.

### Where things are

| Package | What is in it |
|---|---|
| `source` | files and their formats, WAV, SigMF, sockets, radios through pysdrlib (imported only when used); sinks |
| `signal` | `filters` (FIR, IIR, windows, pulses, fractional delay, polyphase channelizer), `sample` (resampling), `channelize`, `DCBlock`, `Deemphasis`, `Conjugate`, `channel` (the simulated medium) |
| `sync` | first class: `time.OerderMeyr`, `phase.MPower`, `equalize.BlockLS`, `CPSync`, tracked by `track`; `FrameSync`, `Delimiter`, `PilotPhase`, `PilotEqualizer`, `Align`, `Puncture`, `Acquire`, `AcquireFEC`, `acquire_code`, `Squelch`, `PeakAGC`, `Burst` (splits a stream into bursts), `BurstTx` (places them), `Preamble` (`attach_tx()`: its `PreambleTx`), `ToneLock`. Second class: `time.MM` (Gardner), `time.TLL`, `Costas`, `AGC`, `equalize.LMS`, `CMA` |
| `wave` | `Shape`/`Matched`, `Stagger`, `FM`/`FMDemod`, `OFDMTx`/`OFDMRx`, `AMDemod`, `SSBDemod` |
| `symbol` | `Constellation` (PSK, QAM, APSK, ASK), `Differential`, `Spread`/`Despread`, `Carriers`, `Header`, `Pilots`, `PhaseScramble` |
| `code` | FEC, interleaving, rate matching (`code.types`), scramblers (`code.scram`), `Line`, `Marker`, `Uncoded` |
| `link` | CRCs, HDLC, `Header` (fields, CRC, payload length) |
| `grid` | resource grids of OFDM systems. `lte`: `LteCell` (PSS/SSS cell search, half-frame timing), `LteGrid` (each subframe's grid with every port's channel), the channels `PBCH`, `PCFICH`, `PDCCH`, `PDSCH`, `tx` (a downlink to test with). `nr` (15 kHz): `NrCell` (cell search over a wide offset, LEO Doppler and clock tracked), `NrGrid`, `PBCH`, `PDCCH`, `PDSCH`, `tx` (a downlink, and `leo`: a satellite pass) |
| `profile` | loading, steps, header-driven control, runners |
| `parallel` | the frame-parallel region (see [Parallelism](#parallelism)) |
| `sequence` | recurrences, Gold sequences, chip codes, words: what scramblers, spreaders and searches share |
| `analysis` | outside the data path: spectra, rates, surveys, blind analysis; can write definitions |
| `base`, `math`, `utils`, `viz` | `Data`, `Stage`, `Pipe`, `Stream`, type tags, control, kernels; finite fields and GF(2); helpers; plots |

Imports run down this list: nothing in the data path imports `analysis`, `profile` or `parallel` (`test_layering.py`; the one
exception is `AcquireFEC`, which imports `analysis.identify` inside the call).

### Config resources

Definitions are organized by **what they define**, not by the package that loads them; any operator references a resource by
name, and a file's `base` and `type` pick the implementation.

| Resource | Holds | Used by |
|---|---|---|
| `constellation` | point sets | symbol |
| `pulse` | shaping pulses, each `"kind": "symbol"` or `"frequency"` | wave |
| `filter` | `fir`, `iir`, `window` | signal |
| `sequence` | `recurrence`, `gold`, `chips`, `set`, `word` (LFSRs, Gold codes, Barker and GPS C/A chips, the CCSDS ASM ...) | code, symbol, sync |
| `scramble` | `additive` (a sequence and its restart) or `feedthrough` (self-synchronising) | code |
| `fec`, `interleave` | `block`, `conv`; `block`, `forney` | code |
| `crc`, `frame` | CRCs; frame formats such as HDLC | link (and code that carries a CRC) |
| `header` | `physical` (known symbols and a short code), `link` (fields, CRC, length) | symbol, link, sync |
| `table` | rows of parameters looked up by header fields (DVB-S2 MODCODs, 802.11b rates) | profile controls |
| `profile` | whole signals | |

One sequence serves everything that needs it: `scramble/additive/lte_pdsch` names `lte_gold` with
`{"c_init": {"expr": "(rnti << 14) + ..."}}`, and `rp.sequence.load("lte_gold", c_init=...)` gives the same sequence to a
spreader or a search.

## Sync methods

**Block estimation and block-rate tracking are first class; per-sample loops are second class.**

- **First class:** estimate from a window with block operations (FFTs, correlation, least squares), then a slow loop run
  **once per window** carries phase and rate across windows and gates outliers. `OerderMeyr` (square-law timing, cubic
  interpolation; follows a clock 200 ppm off), `MPower` (M-th power: FFT acquisition, then per-window phase), `BlockLS`
  (least squares per window against decisions), `CPSync` (OFDM timing and frequency from the cyclic prefix), all tracked by
  `track` (a critically damped second-order loop, wide while it pulls in, coasting through gated windows).
- **Second class:** Costas, PLL, M&M, Gardner, TLL, LMS, CMA, decision-directed loops, for signals where blocks are not enough.

A profile's sync steps name **what** is estimated (`carrier`, `timing`, `equalize`, `frame`) and get the first-class method
(`timing: oerder_meyr`, `carrier: mpower`, `equalize: ls`); a loop is chosen by name in `defaults`:
`{"timing": {"method": "gardner", "alpha": 0.3}}`.

Rules for feedforward estimators:

1. **Windows sit at absolute stream positions** (`[k*W, k*W + L)`, history carried between blocks), so output does not depend
   on how the stream is cut.
2. **Drifting parameters are tracked across windows**, not held per window (a free-running SDR is 10-20 ppm off and wanders),
   and the correction is evaluated per sample.
3. **Windows are set in time, not symbols**: oscillator phase wanders in time (cycle slips at 8 ms windows, none at 2 ms).
4. **Latency is declared** in `stream.est[name]["latency"]` (seconds), not added to `meta.delay`, the group delay.

**`AcquireFEC`** does for the code what `Acquire` does for the modulation: on timed, carrier-corrected symbols it finds the
configured code, bit map and alignment (`analysis.identify`: convolutional codes, RS behind a Forney interleaver, HDLC, block
codes), names the profile that decodes it, then tracks by re-encoding once per window and putting carrier turns and symbol
slips right. With `blind=True` the [blind tools](#blind-analysis-rppsanalysis) run when no definition fits; what they find
is a `Guess`, a definition that decodes once saved.

## Profiles

A profile is a definition of one signal: operators, order, parameters. `config/profile/<family>/<mode>.json`;
`rp.profile.load("ieee80211b/1m")` gives `p.rx()` and `p.tx()` runners (`out = runner(block)`, transmit
`runner(block, fields=...)`).

**One list of steps, in receive order.** A transform step names a pair, runs as written on receive and inverted, in reverse, on
transmit; a sync step is skipped on transmit. Profiles take `params` like any definition; `defaults` holds sync methods and
loop settings. A step that loads a definition names its kind and reference as the loaders take them: `{"code": "conv",
"name": "k7_171_133/3_4"}`, `{"code": "interleave", "name": "forney/12_17"}`, `{"sync": "puncture", "conv": "..."}`,
`{"symbol": "constellation", "name": "psk/qpsk"}`. `python -m rpps steps` lists every kind.

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

builds `rx = carrier + Stagger("i") + halfsine matched + timing + Spread decode` and `tx = Spread encode + Shape(4, halfsine)
+ Stagger("q")`. The carrier comes before `Stagger("i")`: a phase offset mixes I into Q while Q is half a symbol late.

**Bursts:** a `{"sync": "burst"}` step (rpps.sync.Burst's settings) makes a burst demodulator. The steps before it run on
the stream; the steps after it are built anew for every burst and run on it alone (rpps.profile.burst), so every packet is
acquired fresh and nothing a stage, decoder, marker or header holds crosses from one packet to the next. The runner gives a
list, one output per burst that ended in the block, each with its own `stream.est` (`Blocks` flattens it). The burst
floor is the noise windows' own (the lowest cluster of window powers), so packets may fill nearly all the air: a few quiet
windows between them are enough. A
`{"sync": "preamble"}` step after it finds each burst's preamble: by repetition (`repeat`, `window`), or by a waveform the
profile makes itself from a `sequence` word (or `word` + `bits`, after `lead` bits) through its own transmit steps, from the
first that maps symbols to bits back to the preamble's input. On a real stream (an FSK discriminator's output) it matches by
the correlation coefficient and `correct` takes off the level a carrier offset leaves; `{"sync": "timing", "method":
"known", "anchor": <its name>}` places the symbols on what it found. `ble/adv` is built so.

Transmitting, the burst step is the mirror (rpps.profile.burst.BurstTxRunner, rpps.sync.BurstTx): a call sends one burst
through steps built anew for it, and the step places it on one output stream: `gap_s` after the one before or at `at=`,
with `ramp_s` ramps, `freq=` from the centre (or the step's `hop`, cycled) at the stream's rate `Fs`. Bursts on different
carriers may overlap (MF-TDMA). A call gives what no later burst can reach any more: the samples up to its start.
`"attach": true` on a preamble step sends its waveform before each burst (`Preamble.attach_tx()`), for a preamble no later
step sends.

**Branches:** a last step `{"branch": "channels", "order": [...], "frame": N, "branches": {name: [steps]}}` fans the
chain out (rpps.profile.branch): every block of the trunk goes through each branch in `order`, each a chain of its own in a
stream of its own, and the runner gives {name: output}. A header in one branch drives steps in the branches after it: its
fields' position is carried back to the trunk's output and the driven branch takes the change there. With `frame` (trunk
items a frame), the branches read whole frames, so a header rules on a frame before the branches after it read that frame,
however the blocks were cut. A step id is one step across all branches; branches are receive only. Fields published with
a span (a CFI, a DCI: their own subframe) go back to the defaults after it in the driven branches too. A header in the trunk
drives steps in branches too: the trunk ends on an exit step the controls drive in their place, which hands each change on
at the trunk's output position. `dvbs2/mis` is built so: the PLS (trunk) sets each input stream's BBHEADER (a branch, `select`
keeping its ISI). DVB-S2's own PLS-to-payload control needs no branch: the PLS rules its own frame, forward, in one chain.

**Resource grids:** `grid` steps (rpps.grid) take an OFDM system's samples to frames of its resource grid, and its physical
channels read them. `lte/downlink` is built so: `{"grid": "lte_cell"}` finds the cell (PSS, SSS: the cell id, the carrier
offset) and cuts the samples into half-frames from a subframe 0, tracking the sample clock; `{"grid": "lte_grid"}` gives a
frame per subframe, its grid and the channel of ports 0-3 from the reference signals; then a branch per channel, `frame`
one subframe. Each channel publishes what it decoded as a header does: the PBCH the port count (its CRC mask) and the MIB,
the PCFICH its subframe's CFI, the PDCCH a DCI 1A to the SI-RNTI, and the controls carry them to the channels after it,
down to the PDSCH, which gives the transport blocks (SIB1, the SI messages). Mappings the standard gives by formula (where
the reference signals, the REGs and the allocation are) are code in `grid`; tables (transport block sizes) and sequences
(the Gold sequence) are definitions. `nr/downlink` is the same for 5G NR at 15 kHz: `{"grid": "nr_cell"}` searches the PSS over
a wide offset (a LEO pass at S-band moves the carrier by some 50 kHz) and tracks each SS/PBCH block's timing and offset
with their rates (the Doppler sweeps hundreds of Hz a second, the range stretches the samples by tens of ppm); a frame
per slot; branches pbch (MIB, SFN), pdcch (CORESET 0 and the Type0 search space from the MIB; DCI 1_0, polar) and pdsch
(LDPC, SIB1). NR's channels each estimate the channel from their own DM-RS. The phase NR's upconversion gives
each symbol (38.211 5.4) reaches a receiver as e^(-j 2 pi fc t_l) on symbol l (fc: the frequency at 0 Hz, the block's
raster frequency); not every cell applies it (on air one did, one did not), so with `fc` the cell search tells which and
the PDSCH tries both, its CRC choosing.

**Sources that start later in the chain:** the runner skips every receive step before what the source gives (`Source.REP`,
`APPLIED`): timed symbols with the carrier on skip signal, wave and timing; a BBFRAME stream skips everything through code.

**Messages:** every frame a profile receives is also a record, `rp.profile.Message` (rpps.profile.message): its kind (the
step's id), the fields it decoded, its payload, `crc_ok` (None where there is no check), and where it starts in the stream
the runner was given (`sample`, `time`, `freq`). A step that knows where a frame starts and whether it checked out emits it
next to what it publishes, `rp.base.control.emit(meta, op, kind, fields, bits, ok, at)`: link headers (failed frames too),
the HDLC frame step, the DVB-S2 PL header, the LTE and NR channels. A step whose output is a run of fixed-size frames with no
step of their own (TS packets after DVB-S's RS, CCSDS transfer frames) says so in the profile and the runner cuts them:
`{"code": "scramble", ..., "message": {"kind": "packet", "bits": 1504, "check": true, "fields": [{"name": "pid", "from": 11,
"bits": 13}]}}` (`check`: a decoder before it marks its failures, so none marked is good). The runner places each message
through the maps of the steps before it, drops what a re-run span emitted after its checkpoint, and hands each out once from
`runner.take_messages()`; burst and branch runners place their children's in the outer stream. A parallel region's kernels
decode inside: each frame result is its message (`"frame"`, crc_ok False when dropped), without the header fields a serial
runner gives.

**Only some kinds:** `profile.kinds()` gives `{kind: step id}` (an operator that emits says so with `EMITS = True`; a
`"message"` cut's kind is its own), `rp.profile.info(name).kinds` the same list for a menu. `profile.rx(kinds={"pbch"})`
(also `Decoder(..., kinds=)`, `rpps rx --kinds pbch,pcfich`) hands out only those, and builds only what they need
(`profile.needed(kinds)`): their steps, and every header whose fields drive a needed step, and so on (LTE's PDSCH needs the
PDCCH, which needs the PCFICH, which needs the PBCH). A branch none of whose steps is needed is not built; a chain (the trunk,
a branch, the steps of a burst) ends at its last needed step, so the runner's output is that step's (DVB-S2's `pls` alone:
no descrambling, LDPC or BCH). A parallel region runs whole, or not at all when none of it is needed; with one, its
`"frame"` is a kind (`kinds(parallel=True)`) in place of those its steps emit. A kind the profile doesn't give is a
ConfigError.

**What a profile needs:** `"requires"`, beside `params`, as strings over the params and `baud`, evaluated in order (each may
use those before): `baud` (symbols, chips or subcarrier spacing; null where the standard leaves the rate open), `Fs`, `bw`
(the channel taken), `min_samples` (enough for one message); `input` ("audio": an FM channel's audio, AX.25), `from_cf`
(params the carrier gives: `{"channel": "37 if abs(cf - 2402e6) < 1e6 else ..."}`). Strings, not `{"expr": ...}`: they are
evaluated when asked (`rp.profile.info(name, baud=, **params)`, `catalogue()`), not when the profile loads. `rp.profile.Decoder`
(rpps.profile.decoder) reads them to bring a selected signal to the profile: shift and staged decimation, an FM demodulator
for audio, a fractional resampler to `Fs`; a rate left open is given or measured (`Lock.find`) first.

### Wave pairs

| Step | Receive (as written) | Transmit (inverted) |
|---|---|---|
| `shape`, symbol pulse (rrc, rc, halfsine, rect) | `wave.Matched(pulse)`, unit gain at the instants | `wave.Shape(sps, pulse)`, polyphase |
| `shape`, frequency pulse (gaussian, FSK rect) | none (the `fm` discriminator) | `wave.Shape(sps, pulse)` |
| `stagger` | `Stagger("i", delay)` | `Stagger("q", delay)` |
| `fm` | `FMDemod` + integrate over the symbol | `FM(h, step)` |
| `ofdm` | remove CP + FFT | IFFT + CP |

Whether a pulse is a symbol or a frequency pulse is the pulse definition's `kind`, not the profile's.

### Header-driven control

Headers set what follows: DVB-S2 PLS (modulation, LDPC rate, pilots, frame length, per frame), the 802.11b PLCP (payload
rate, mid-packet), length fields (802.15.4, BLE, ADS-B), the P25 NID, adaptive links. The change has to land at the right
place without losing lock, so a **control path** runs back up the chain (`rpps.base.control`, `rpps.profile.control`,
`rpps.profile.runner`):

1. **Publish with a position.** A step calls `rp.base.control.publish(meta, op, fields, at=, out=, span=)`: `at` is the
   absolute input position the fields apply from, `out` where that falls in this call's output, `span` how long they hold. A
   frame sync publishes `{"frame": k}` at each start. A step that drops or inserts items reports
   `rp.base.control.anchor(meta, op, at, out)`. Everything lands in `stream.control` (`log`, `fields(step)`, `reports`,
   `rerun`, `buffer`).
2. **Map fields to parameters.** `controls`, keyed by the publishing step's id, says which steps take which parameters, usually
   through a table:

   ```json
   "controls": {
     "frame": {"drive": {"pls": {"frame": "frame"}}},
     "pls": {"table": "dvbs2/modcod", "key": ["modcod", "short"],
             "default": {"modcod": {"param": "modcod"}, "short": 0, "pilots": {"param": "pilots"}},
             "frame": {"calc": "kbch - 80"},
             "drive": {"map": {"name": "constellation"}, "ldpc": {"name": "ldpc"}, "pilots": {"slots": "slots", "pilots": "pilots"},
                       "plscram": {"length": {"calc": "slots*90 + pilots*((slots - 1)//16)*36"}}}}}
   ```

   A source is a field, a column of the row, or `{"calc": ...}` (worked out when the fields arrive; `{"expr": ...}` is load
   time). The loader refuses unknown steps or parameters, ambiguous ids, and parameters a step does not declare.
3. **Change at a position.** Parameters are fixed unless controlled. Operators declare `CONTROLS` per parameter: **live**
   (changed in place, state kept: constellation, puncturing, LDPC rate, frame length; taken through `control(**params)`),
   **restart** (rebuilt from the position, what was in flight dropped and reported: interleaver depth, a Viterbi's
   constraint length, FFT size), or not controllable (a tracker's own estimates). A live change keeps lock: QPSK to BPSK
   changes the constellation and the carrier estimator's power, not the timing. A change is a frame start for the step; what
   the last frame left in it is dropped and reported, never joined to the next.
4. **Positions translate** between representations through each step's rate and delay; that is why every block carries
   its absolute position. A change for a later step travels with the data like a tag (input split at the position).
5. **Late changes re-run.** A change for a step before its publisher (a link header changing its own payload's decoding)
   is carried back through the position maps, and the runner re-runs from the earliest such step out of a re-run buffer,
   in pieces checkpointed at least the controls' `reach` apart. The buffer follows the current frame length and rates. A
   change whose data has left the buffer is not applied backwards: the frames are dropped and reported, and processing
   continues with the change. Standards that announce ahead (DVB-S2) never re-run.

**Transmit:** each frame's fields come from the caller (`fields=`, one dict or one per frame), else the entry's `schedule`,
else its defaults; `frame` gives the items one frame takes. A header step that publishes drives the steps after it (the
802.11b PLCP builder switches the rate where its payload starts).

## Parallelism

Real time needs more than one core, and later a GPU or FPGA. Once frame sync has found a frame and its header has set the
parameters, the rest is per-frame work with no state across frames. A profile marks where that region starts
(`"parallel": "<step id>"`); `rpps.profile.parallel` refuses a region with stream-scoped steps (`SCOPE = STREAM`), a
publishing header or a late change.

The unit is the **batch**, not the thread: N frames through one program pays on every target.

```
profile.rx(parallel=rp.parallel.Parallel("thread", workers=4))   -> ParallelRunner
  front       the ordinary Runner, stopped at the region; the changes it resolved cut the input at frame starts
  Scheduler   frames -> batches of one key -> Executor (inline, threads by rows, processes, later GPU / FPGA) -> results in order
  BatchPlan   per key (the header's parameters, the frame's shape): the region's kernels, shapes checked, describe()
  Kernel      a pure function over [N, L] arrays, run(xp, batch); LoopKernel runs frame by frame through the operators
```

- An operator gives its kernel with `op.kernel(params, how)`; a step without one runs through `LoopKernel`, which is also the
  oracle every kernel is checked against (`test_batch_kernels.py`). Kernels: `symbol.kernels` (`Demap`, `Rotate`, `Gather`),
  `code.kernels` (`BlockDecode`, `XorMask`), `sync.kernels` (`PilotLine`); the protocol is `base.kernel`.
- Kernels belong to generic operators, never a protocol; they use only what numpy and cupy share (`xp`), hold their
  constants, keep shapes fixed per key, and mark a failed row instead of raising. `BatchPlan.describe()` is data, what a
  device library or HLS generator would build from.
- Processes use `fork` where available, else `spawn` (scripts need a `__main__` guard); each builds its region from a
  picklable `RegionSteps`.
- On `dvbs2/normal` the output equals the serial runner's bit for bit, for any block cut, worker count and batch size.
  Measure with `python tests/parallel_bench.py 64 4`.

## Core abstractions and streaming rules

**Data** (`base/data.py`) wraps an array with its `Meta`: `Fs`, `sps`, centre frequency, group delay, stream state, and
`meta.pos`, the absolute position of its first item (`meta.time_of(k)` gives source time). Nothing needs setting up or
resetting.

| Operator | Means | On |
|---|---|---|
| `data * pipe`, `data / pipe` | encode / decode | `Pipe` (invertible transform) |
| `data + stage` | process | `Stage`, `BitStage` |
| `pipe * pipe`, `stage + stage` | a reusable `Pipeline` | either |
| `data >> sink` | write out | `Sink` |

A `Stage` keeps its settings on the object and everything that carries between blocks in `new_state(meta)`, held by the
stream; `step(samples, st, meta)` processes a block and sets `meta.op`. `SCOPE` is `STREAM` (state crosses frames) or `FRAME`.
`Stream` holds states, estimates (`stream.est[name]`), detections (`det`), control (`stream.control`) and load statistics.

Streaming rules (tested by cutting the same input differently: `test_stream.py`, `test_position.py`, per-stage tests):

1. **Any block length, including 0**; never drop or repeat items across blocks.
2. **State lives in the stream state**, so one stage can serve several streams.
3. **Whole blocks** through `rp.base.rpps.align(ctx, key, x, block, pad)`: the leftover waits, a flush pads (encode) or
   drops and reports (decode).
4. **Positions are absolute**; a stage that drops or inserts items anchors it.
5. **Feedforward estimators** follow the rules in [Sync methods](#sync-methods).
6. **Soft information flows down**; sync on channel bits passes LLRs through.

## Definitions (config)

Codes, scramblers, constellations, pulses, filters, sequences, headers, tables and profiles are JSON, not code: a new instance
is a new file, a new kind is a new loader. Format, naming and the keys each loader takes: [config/README.md](config/README.md).

- `config/<resource>/[<group>/]<name>.json`; `base` picks the loader, `type` the implementation. Search order:
  `rp.add_config_path()` and `$RPPS_CONFIG`, then the shipped `config/` (`rpps/config` in a wheel, copied by `setup.py`;
  CI checks the wheel has it).
- `params` with `{"param": ...}` and `{"expr": ...}` values, evaluated at load. `"extends"` makes a child (dicts merge, lists
  replace, null deletes); `"abstract": true` parents do not load alone. A load-time choice is a param; a variant a standard
  names is a child.
- `"tags"` (declared in `config/tags.json`) name the standards using a definition: `rp.tagged("dvb-s2")`, `rpps ls --tag`.
  No aliases: one name per definition, one reference per loader (`rp.code.load("conv/k7_171_133/3_4")`,
  `rp.link.load("crc/crc32_mpeg2")`).
- Loaders register per resource (`_config.Domain.register`); `ls`, `find` and `load` then work on it.

## Effort

`rp.set_effort("fast" | "balanced" | "max")`, `with rp.effort(...)`, `RPPS_EFFORT` or `rpps rx --effort` trade speed for
recovered SNR. Stages look the level up when they run; `balanced` is what the definitions say (`rpps._effort.SETTINGS`).

| Setting | fast | max | Measured (max against balanced) |
|---|---|---|---|
| LDPC / turbo iterations | x0.5 | x4, stopping early | LDPC 1/2 at 0.9 dB: FER 3/16 → 0/16 at 2x time; LTE turbo at 0.2 dB: 6/12 → 0/12 |
| Max-log or exact (demapper, BCJR, turbo) | max-log | exact, no extrinsic scaling | turbo log-MAP 0.1-0.2 dB, 5-7x time |
| Timing interpolation | cubic | 12-tap windowed sinc (balanced: 8 taps below 4 sps) | EVM floor at 2 sps: cubic -26 dB, 8 taps -48, 12 taps -55 (pulse truncation -56) |

Other decoding behaviour worth knowing:

- LDPC is layered normalized min-sum (checks coloured once per code), about half the iterations of flooding.
- Soft values go as far as something uses them: additive descrambling flips LLR signs, Forney deinterleaving moves LLRs, BCJR
  hands on a-posteriori LLRs (RS uses them for erasures, keeping two checks spare), differential decoding is the product
  detector `y conj(y[n-1])`.
- A decoder returns its status, it never keeps it: an implementation's `_decode` gives (output, errors per codeword or
  None) and `decode(x, status=True)` passes both on, so one loaded (cached) code decodes on several threads at once.
  Soft or path outputs are methods that return them too (`app` of BCJR and turbo, `paths` of fsm), never attributes. A code
  step puts the errors on its output (`data.errors`, not carried further) and sets `data.failed` from them (the last code
  with a verdict decides); later steps carry `failed`, and a link header drops a frame with any failed item whatever its
  CRC says. The parallel region drops such rows.
- Under 4 sps Oerder & Meyr takes its square-law line at twice the rate (at 2 sps it aliased onto its mirror).

## Native kernels

`rpps._native` (C++17, pybind11, `csrc/rpps_native/*.cpp`) holds the per-sample loops and decoders (AGC, Costas, timing,
equalizers, IIR, Viterbi, BCJR, LDPC, polar list decoding, RS, BCH, demapping, resampling, FIR decimation, carrier and
frame sync) and the blind
tools' slow loops (`analysis.cpp`). Without a compiler the build warns and RPPS runs its Python loops.

**The Python loop is the reference.** Every kernel's Python version stays in the stage that calls it, and tests compare the
two (`test_native*.py`, `test_parallel.py`, `test_batch_kernels.py`): a kernel changes speed, never results beyond rounding.
Stages look `rpps._accel.native` up when they run, so tests can switch it off.

| Variable | Effect |
|---|---|
| `RPPS_NO_NATIVE=1` | never load the kernels |
| `RPPS_REQUIRE_NATIVE=1` | a failed native build is an error (CI and wheels) |
| `RPPS_THREADS=n` | threads for kernels and executors that spread work |
| `RPPS_CONFIG` | extra config directories |

Adding one: write `csrc/rpps_native/<name>.cpp` with `register_<name>(m)`, declare it in `common.hpp`, call it from
`module.cpp` (`setup.py` globs `*.cpp`); keep the Python fallback behind `_accel.native`; release the GIL for anything long;
test it against the Python loop.

Pitfalls found the hard way:
- **`long long` loop and tap indices.** CPython builds with `-fno-strict-overflow`, so `int` indices are not widened: 3x
  slower inside the module.
- **`target_clones("avx2", "default")` only**, behind `#if` for GCC (MSVC has neither it nor `__builtin_*`; `common.hpp` has
  portable helpers such as `ctz64`). `"arch=haswell"` made the 4-accumulator dot product 40% slower (FMA lengthens the chain).
- **Size `parallel_for` chunks by work** (about 128k multiply-adds), not items; the pool is persistent and reset after a fork.
- **Keep BLAS products out of per-block paths:** OpenBLAS spins between calls and slows everything around it
  (`OPENBLAS_NUM_THREADS=1` shows it).
- **Benchmark best-of-N in one process** (`tests/bench_stages.py`, `parallel_bench.py`, `resample_bench.py`).

## Testing

```
python -m pytest -n auto -m "not slow"                       # the fast tests, on every core (CI runs this)
RPPS_NO_NATIVE=1 python -m pytest -n auto -m "not slow and not kernels"   # the Python loops
python -m pytest -m oracle                                   # the scipy comparisons (needs .[oracle])
python -m pytest tests/unit/test_remez.py                    # one file
```

| Marker | Meaning |
|---|---|
| `slow` | more than a few seconds even with the kernels; not run in CI |
| `kernels` | over a minute without the kernels; left out of the no-kernel run |
| `oracle` | compares against scipy through `pytest.importorskip`; skipped without it |

- **RPPS must not need scipy**; nothing under `src/` may import it (`test_numpy_only.py`).
- `conftest.py` sets the Agg backend and gives `rng` (seeded), `bits`, `awgn`, `shaped`, `cpfsk`.
- `test_contracts.py` runs every definition against its category's contract, so new definitions are covered automatically.
- A statistical test asserts with a margin measured over many seeds, not one lucky seed.
- `projects/tests` hold the worked examples' tests; they need no radio. Run them on their own (`pytest projects/tests`).

## Adding things

- **A definition** (constellation, code, scrambler, pulse, CRC, sequence, header, table, profile): a JSON file in
  `config/<resource>/` with `base`, `type`, `comment` and `ref`; `python -m rpps ls <resource>` to see it,
  `python -m rpps fmt --check` for style.
- **An implementation** (a code type, a sync method): put it in its category's package ([the rule](#the-rule)); subclass
  `Pipe`, `Stage` or `BitStage` with `CATEGORY`, `takes` and `SCOPE`; register its loader and its profile step
  (`rp.profile.steps.register_step`, `register_sync_method`), or name any `Stage` by module path in a profile
  (`{"signal": "stage", "class": "mypkg.Gain", "args": {...}}`). A code type subclasses `code.types._code._code` and defines
  `_decode(x, ...) -> (output, errors per codeword or None)`, keeping nothing of the call on itself. Write Python first
  with a block-cut test; move hot loops to C++ after.
- **A sync estimator:** block operations over windows at absolute positions, tracked by `sync.track`. A per-sample loop is
  second class, chosen by name in `defaults`.
- **A signal:** a profile, which should need no new operator code ([Profiles](#profiles)).
- **A source:** subclass `Source`, say what representation and sync it gives, set its rates and each block's position.
- **Branches:** `data + rp.Tee(a=..., b=...)` runs each branch in its own stream (FM stereo L+R and L-R); profiles stay one chain.

## CI and releases

| Workflow | When | Does |
|---|---|---|
| `tests.yml` | push to `main` / `dev`, pull requests | fast tests on Python 3.12-3.14, Ubuntu and Windows (`-n auto`; Windows 30 min, Ubuntu 20); the oracle tests on 3.12; a no-kernel job; sdist → wheel and the examples in a clean venv; the docs build |
| `deploy-docs.yml` | push to `main` | pdoc to GitHub Pages |
| `python-publish.yml` | a GitHub release | cibuildwheel (CPython 3.12-3.14; Linux x86_64 and arm64, Windows; each wheel must carry `rpps._native`), checks the tag against the version, publishes by trusted publishing |

To release: move CHANGELOG.md's "Unreleased" under the new version (every rename or removal under **Breaking**, with
what replaces it); `python builder.py -v p` (or `m`, `M`) previews the bump, `-r` makes it (edits `pyproject.toml` and
`src/rpps/__init__.py`, commits, does not tag); merge to `main`; publish a GitHub release tagged exactly `v<version>`.

## Conventions

- **Operators first:** keep features inside `*`, `/`, `+`, `>>` rather than adding setup calls.
- **No aliases:** a rename or move drops the old name and fixes the callers.
- **Be explicit:** fix a caller rather than loosen a contract check.
- **Category by what it takes and gives** ([the rule](#the-rule)).
- **Read the standard** when a decode does not work, instead of guessing.
- **Examples live in `examples/` and `projects/`**, scratch stays uncommitted; reuse RPPS so useful parts can move
  into the package.

## Not yet built

Reviewed 2026-11-19 against errorcorrectionzoo.org's realized codes, AFF3CT, liquid-dsp and gr-satellites.

- **Codes:** NR polar parity-check bits (an uplink message of 12-19 bits), LDPC limited-buffer rate matching from a
  transport block's TBS_LBRM, HARQ combining across redundancy versions; LDPC tables for CCSDS AR4JA / C2, DVB-S2X, DVB-T2,
  ATSC 3.0; UMTS and DVB-RCS duo-binary turbo; 4B5B, 8b/10b; Fletcher and Internet checksums; fountain codes.
- **Waves and sync:** coherent CPM (GMSK, SOQPSK) in the data path (offline only: `analysis.modulation.cpm`); VSB / PAM, chirp
  spread spectrum, space-time codes; a burst step in a parallel region; a channelized burst step (receiving MF-TDMA or a hopping carrier); a preamble `offset`
  with known timing `anchor` (the grid lands off); the transmit side of branches (and of `grid`:
  `grid.lte.tx`, `grid.nr.tx` are test transmitters); NR at 30 kHz and above, NR VRB-to-PRB interleaving, several
  code blocks a transport block, multiple antenna ports; LTE beyond SIB1 (unicast DCI formats, multiple code blocks, TDD, extended prefix); `lte/downlink` in
  real time at 15.36 MS/s (about 1.1x slower; `nr/downlink` keeps up at 7.68 MS/s); a first-class carrier estimator for
  staggered signals, fractionally spaced block equalization, DFE; the DVB-S sync bytes, 802.11b SFD and DVB-S2 SOF as
  `sequence/word`; matched filter fused with Oerder & Meyr.
- **Links, sources, profiles:** PPP, KISS, AX.25 addresses, MPEG-TS, CCSDS TM / AOS frames and space packets, FX.25,
  block-coded headers; tun/tap, audio devices, transmitting to a radio; profiles for DMR, P25, 802.11a/g/n, ATSC, GPS C/A,
  DAB, UMTS, CCSDS turbo (and `projects/` using the BLE, ADS-B and 802.11b profiles that exist); `dvbs2/normal` dummy frames;
  positions and gaps in the parallel runner's output; GPU and FPGA backends.

## Blind analysis (rpps.analysis)

Tools for signals no definition covers. Each states its statistic and chance level, so "nothing found" is an answer; each
module's docstring has the method, reference and limits. Numpy only, outside the data path. Most take baseband at a few times
the signal's width (some noise floor beside it), locked symbols, or bits. `identify.solve` peels layers off a decoder's output
in turn; structure that holds at every alignment (constant bits, idle words) is not taken for a code. Rank methods need about
n + 48 clean words of n bits; strong codes at low SNR are beyond them on channel bits.

**What is it, and at what rate**

| Question | Tool | How (and limits) |
|---|---|---|
| symbol rate, carrier order and offset | `spectrum.cyclic.features` | lines of `\|x\|^2`, `x^M` and the discriminator against a median floor, calibrated false alarms; rate by harmonic comb |
| every cycle frequency (signals sharing a band) | `spectrum.scf.cycle_frequencies`, `coherence` | FFT-accumulation spectral coherence, conjugate too |
| which constellation | `modulation.cumulants.classify` | C40 ... C84 against each constellation's; orders within QAM / ASK need the SNR |
| FSK / MSK / GMSK, CPM index | `modulation.fsk.analyze`, `cpm_index` | instantaneous frequency; h from the phase lattice |
| GMSK / SOQPSK bits, which pulse | `modulation.cpm.demodulate`, `identify` | per-survivor Viterbi, each survivor its own carrier loop |
| OFDM pilot pattern | `spectrum.ofdm_pilots.discover`, `grid` | grid from CP peaks; period from coherent repetition; pilot lattice, boost, empty cells |
| TDMA bursts, frame and slots | `spectrum.burst.demodulate` | each burst on its shared preamble, data-aided carrier and phase |
| two co-channel signals of equal power | `spectrum.separate.estimate`, `separate` | each one's carrier, timing and phase; joint sequence estimate |
| which named system | `catalogue.match`, `features` | measured features against systems from their standards, weighted |
| PM, residual carrier, subcarriers | `modulation.pm.analyze`, `subcarrier`, `nrz` | carrier tracked, phase around it |
| DSSS | `sequence.spread.period`, `chip_rate`, `code`, `despread` | windowed autocorrelation variance; code from eigenvectors |
| frequency hopping | `spectrum.hopping.track`, `dehop` | short-time spectrum clustered into channels and dwells |
| kind of waveform | `spectrum.waveform.classify` | tones, OFDM numerology against known systems, bursts, envelope kurtosis |
| tones in a short record, neighbours | `spectrum.pencil.lines`, `modulation.impairments.neighbours` | matrix pencil; bands split at valleys and level steps |

**Is the lock good, and what is left**

| Question | Tool | How |
|---|---|---|
| can hard decisions be trusted | `modulation.quality.gate` | rotation, drift, near-boundary decisions against the measured Es/N0 |
| how much is locked; what is underneath | `modulation.cancel.cancel`, `align` | regenerate, subtract, residual against the floor |
| amplifier, IQ, phase noise | `modulation.impairments.amplifier`, `iq`, `phase_noise` | ring gains and turns; image ratio; phase error from neighbours |
| separate an interferer | `spectrum.scf.fresh` | least-squares FRESH filter trained on a reference (not blind) |

**Which code, scrambler, interleaver**

| Question | Tool | How |
|---|---|---|
| convolutional code (rate 1/n, K) | `code.conv.mother.scan_conv`, `scan_defined_conv` | generator-pair parity relation against chance; the configured codes by re-encoding |
| puncture of a known mother code | `code.conv.puncture.search` | every rate P/(P+1) pattern through the Viterbi decoder |
| Reed-Solomon over GF(256) | `code.block.rs.search` | shared roots by Galois-field Fourier transform at every byte phase |
| binary block code | `code.block.linear.solve_block` | length from rank deficiency, checks from the null space; then cyclic / BCH, product or LDPC |
| additive scrambler | `sequence.scrambler.additive_search`, `explain`, `search` | Berlekamp-Massey over windows; polynomials factored into known scramblers |
| scrambler taps | `sequence.recurrence.search`, `lfsr`, `keystream` | every recurrence to degree 24 at once; any degree by Berlekamp-Massey |
| trellis code on PSK labels | `code.conv.tcm.search` | Walsh-Hadamard over label windows, every rotation and labelling |
| turbo constituent and interleaver | `code.conv.rsc.constituent`, `turbo_interleaver`, `turbo_interleaver_soft` | 2-bit-label check; permutation from clean or noisy frames |
| LDPC (IRA, QC, large) | `code.block.ldpc.staircase`, `qc_staircase`, `qc_dual`, `qc_size`, `progressive` | one GF(2) solve per structure (a DVB-S2 table from k/360 + 12 clean frames) |
| cyclic / BCH roots | `code.block.ldpc.roots` | zero counts at every beta^j against chance |
| polar frozen bits | `code.block.polar.frozen`, `recognize`, `search` | per-position likelihood over frames (rate 1/2 at -2 dB from 200 frames) |
| interleaver | `code.interleave.block_search`, `forney_search`, `column_delays` | rank deficiency for the size; candidates scored by the code coming back |

**What the bits are**

| Question | Tool | How |
|---|---|---|
| frame period and unique word on symbols | `framing.frames.find`, `cut`, `bursts` | coherence per position over candidate periods |
| sync markers, frame length, HDLC | `framing.bits.find_sync`, `frame_length`, `scan_hdlc` | hits at a period, autocorrelation, flags and FCS against chance |
| PM telemetry end to end | `framing.telemetry.analyze` | BPSK per subcarrier, line code, frame period, counters |
| framing in decoded bits | `framing.payload.signatures` | E1/T1, ATM, GFP, HDLC, IPv4, CCSDS ASM, MPEG-TS, fixed bits, each against chance |
| structured, scrambled or random | `framing.payload.verdict` | signatures, bias, linear complexity, compressibility |
