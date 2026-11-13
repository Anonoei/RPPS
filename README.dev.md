# RPPS development guide

How the project is designed, organized and implemented, and how to add to it. [README.md](README.md) is for users.

- [RPPS development guide](#rpps-development-guide)
  - [Setup and workflow](#setup-and-workflow)
  - [Repository layout](#repository-layout)
  - [Design](#design)
    - [The model](#the-model)
    - [The rule](#the-rule)
    - [Categories](#categories)
      - [Two kinds of framing](#two-kinds-of-framing)
      - [A receive chain, in order](#a-receive-chain-in-order)
      - [Contracts](#contracts)
    - [Sync methods](#sync-methods)
    - [Profiles](#profiles)
      - [Example: IEEE 802.15.4, 2.4 GHz](#example-ieee-802154-24-ghz)
      - [Wave pairs](#wave-pairs)
      - [Header-driven control](#header-driven-control)
      - [Sources that start later in the chain](#sources-that-start-later-in-the-chain)
      - [Parallelism](#parallelism)
    - [Grey areas](#grey-areas)
    - [Where things are](#where-things-are)
    - [Config resources](#config-resources)
    - [Type tags](#type-tags)
    - [Status](#status)
  - [Core abstractions](#core-abstractions)
  - [Streaming rules](#streaming-rules)
  - [Definitions (config)](#definitions-config)
  - [Native kernels](#native-kernels)
  - [Parallel runner](#parallel-runner)
  - [Testing](#testing)
  - [Adding things](#adding-things)
  - [CI and releases](#ci-and-releases)
  - [Conventions](#conventions)

## Setup and workflow

```
git clone https://github.com/Anonoei/RPPS && cd RPPS
python builder.py -s        # pip install -e ".[dev]": the package, its extras and the dev tools
python builder.py -c        # build the C++ kernels in place (needs a C++17 compiler)
python builder.py -t        # pytest
```

`pyproject.toml` is the one place that lists dependencies and versions. Nothing else (CI included) names a package.

| Where | What |
|---|---|
| `[build-system].requires` | setuptools, numpy, pybind11: what building needs (pip installs them in an isolated environment) |
| `dependencies` | what running needs: numpy, pysdrlib |
| `[project.optional-dependencies]` | `viz` (matplotlib), `test` (pytest), `oracle` (scipy), `docs` (pdoc3), `dev` (all of those plus twine, build, bumpver) |
| `[tool.cibuildwheel]` | wheel targets and the wheel smoke test |
| `[tool.bumpver]` | version bump (`tag = false`: tags come from GitHub releases) |
| `[tool.pytest.ini_options]` | test paths and markers |

`builder.py` runs the common steps (`-h` lists them): `-s` setup, `-c` compile, `-t` test, `-d` docs, `-b` build sdist and wheel
(then `twine check`), `-l` install locally, `-v M|m|p` bump the version (a dry run without `-r`), `-u` upload (TestPyPI
without `-r`).

Work on a feature branch and merge into `dev`; `main` is released. Scratch and analysis scripts go in `tests/dev` (or
`scripts/` when they should reuse RPPS components and might be upstreamed).

## Repository layout

```
src/rpps/            the package (one sub-package per category, below)
csrc/rpps_native/    C++ kernels, built as rpps._native (optional)
config/              JSON definitions, shipped inside the wheel (rpps/config)
pdoc_templates/      the docs theme
tests/unit/          the test suite (pytest)
tests/dev/, prop/    exploratory scripts and property experiments, not run by CI
tests/*.py           examples and benchmarks (example.py, bench_stages.py, parallel_bench.py)
projects/            worked examples on real signals (FM, ADS-B, ATSC, LTE, P25...), each with its own README
data/                captures used by tests/dev and projects (git-ignored)
builder.py           local build helper
setup.py             builds the native module and copies config/ into the wheel
```

What each package holds, by category, is in [Where things are](#where-things-are).

`python -m rpps ls` lists every definition; `python -m rpps fmt` rewrites config files in the canonical style.

## Design

RPPS is meant to stay signal agnostic: any implementation of any function should fit, and a new kind of signal should be new
configs plus existing operators. This document defines the categories that make that possible, says where every current
module belongs. The [Status](#status) section says what is built.

### The model

A receiver moves data through a fixed series of **representations**:

```
samples  ->  symbols  ->  channel bits  ->  data bits  ->  packets / messages
```

- **Channel bits** are the bits as they go over the channel: after FEC, interleaving and scrambling, with any sync markers.
  On receive they are LLRs until the code transform decides them.
- **Data bits** are what the link layer carries: decoded, descrambled, aligned.

Three kinds of operation act on them:

1. **Transforms** turn one representation into the next, and come as an invertible pair (transmit and receive). There is one
   transform per step: wave, symbol, code and link.
2. **Conditioning** (`signal`) changes samples without needing to know what signal they carry.
3. **Sync** estimates something unknown about the data and corrects or aligns it. It can sit at **any** representation:
   carrier and timing on samples, phase on symbols, codeword boundaries on symbols or channel bits, packet starts on data bits.

Sync is not one rung of a ladder. A receiver uses it again and again, each time at the point where the thing it corrects
can still be separated from everything else.

A **profile** puts these together into one signal: the transforms, the sync steps between them, and their parameters,
for both directions.

Data flows down the chain; **control** flows back up it. A header decoded late in the receive chain (a link header, a
physical header) can change the parameters of steps earlier in the chain, such as the code rate or the modulation, from a
known position in the stream onward, without losing lock (see [Header-driven control](#header-driven-control)).

### The rule

A function belongs to the category named by **what it takes and what it gives**, not by what a standard calls it.

- If it changes the representation, it is the transform for that step.
- If it estimates something unknown and corrects or aligns the data with it, it is sync, whatever representation it works on.
- If it changes samples and needs no knowledge of the signal, it is signal.
- Inserting or removing known structure belongs to the transform of the representation where that structure lives: pilots
  and physical headers among symbols (symbol), sync markers among channel bits (code), headers and CRC among data bits
  (link). Finding where the structure starts is sync.

Two implementations of the same function share a signature, so they always land in the same category. Anything that obeys
a category's contract is a valid implementation, whatever standard it comes from.

### Categories

Receive direction. Transmit runs the transforms in reverse; sync has no transmit side.

| Category | Package | Takes → gives | Operator |
|---|---|---|---|
| Source | `source` | outside world ↔ samples | `Source(n)`, `block >> sink` |
| Conditioning | `signal` | samples → samples | Stage, `+` |
| Sync | `sync` | samples, symbols or bits → the same, corrected or aligned. Timing recovery also takes samples to symbols | Stage, `+`, one-way |
| Wave transform | `wave` | samples ↔ symbols | Stage pairs |
| Symbol transform | `symbol` | symbols ↔ channel bits (LLRs on receive) | Pipe, `*` and `/` |
| Code transform | `code` | channel bits ↔ data bits | Pipe, `*` and `/` |
| Link transform | `link` | data bits ↔ packets and messages | Pipe, `*` and `/` |
| Profile | `profile` | one whole signal, both directions | builds the pipelines above |

#### Two kinds of framing

What standards call "framing" is two different things, and they sit on opposite sides of the FEC decoder:

| | Lives among | Purpose | Found by | Inserted / removed by |
|---|---|---|---|---|
| Sync markers, preambles, start-of-frame words (CCSDS ASM, DVB-S sync bytes, DVB-S2 SOF, 802.15.4 SHR) | symbols or channel bits, before decoding | let the receiver find codeword and frame boundaries | `sync` | `symbol` (among symbols) or `code` (among channel bits) |
| Link framing (HDLC flags and stuffing, headers, length fields, CRC, packets) | data bits, after decoding | carry the payload | `sync` when the stream is unaligned (HDLC flags), otherwise implied by the code's blocks | `link` |

#### A receive chain, in order

Sync appears between the transforms, not in one place. DVB-S2:

| Step | Category | Representation after it |
|---|---|---|
| channelize, decimate | signal | samples |
| AGC, coarse frequency | sync | samples |
| matched filter | wave | samples |
| timing recovery (picks one sample per symbol) | sync | symbols |
| start of frame (SOF correlation), fine frequency and phase from pilots | sync | symbols |
| PL descrambling, header and pilot removal, demapping | symbol | channel bits (LLRs) |
| bit deinterleaving, LDPC, BCH, BB descrambling | code | data bits |
| BBHEADER, user packets | link | packets |

DVB-S shows sync acting on channel bits, between two parts of the code transform:

| Step | Category | Representation after it |
|---|---|---|
| ... timing and carrier ... | sync | symbols |
| demapping | symbol | channel bits |
| Viterbi | code | partly decoded bits |
| byte alignment from repeating sync bytes, polarity | sync | aligned bits |
| Forney deinterleaving, Reed-Solomon, energy dispersal | code | data bits |
| MPEG-TS packets | link | packets |

#### Contracts

**source**
- Produces blocks at **any representation**, and says which: samples from a radio or a capture, timed symbols from another
  demodulator's ring or a symbol file, data bits, packets. It also says which sync has already been applied to them (for
  example: timed, carrier not removed). Sinks consume any representation (`>> sink`).
- Sets the rates it knows (`Fs`, `cf`, symbol rate) and the absolute position of each block.
- Files, sockets and radios (through pysdrlib) are implementations.

**signal**
- Takes `SAMPLES` or `REAL`, gives the same kind.
- May change `Fs` (resampling, decimation); never needs `sps`.
- Has no transmit/receive distinction: the same filter works both ways.
- Includes `signal.channel`, the simulated medium (AWGN, frequency and timing offsets, multipath, fading), which tests use
  as the transmit-side counterpart of sync.

**sync**
- Takes samples, symbols, channel bits or data bits, and gives the same representation corrected (frequency, phase, gain,
  channel) or aligned (to the symbol clock, a codeword or a packet).
- One exception changes representation: **timing recovery** is the sampler, taking `SAMPLES` at `sps` > 1 to `SYMBOLS`.
  The matched filter before it is wave; deciding the sampling instant is sync.
- Its estimates (frequency offset, timing, lock, SNR, detected modulation, frame start, polarity) go to the stream
  (`meta.stream`, `det`), never into the data.
- Has no inverse and no transmit side.
- See [Sync methods](#sync-methods) for how it estimates.

**wave**
- Transmit takes `SYMBOLS` and raises `sps` (pulse shaping, FM/CPM, stagger, IFFT plus cyclic prefix).
- Receive brings samples back to the form the sampler needs (matched filter, FM discriminator, rail realignment, cyclic prefix
  removal plus FFT). It does not choose the sampling instant.
- Pieces come in pairs: shaping and matched filter, `Stagger("q")` and `Stagger("i")`, `FM` and `FMDemod`. A profile names
  the pair, and the profile loader builds each half (see [Wave pairs](#wave-pairs)).

**symbol**
- Encode takes channel bits and gives symbols.
- Decode takes `SYMBOLS` and gives `SYMBOLS` (differential decoding, despreading, removing pilots and headers, symbol
  descrambling) or `LLR` (demapping). The last symbol step of a receiver always gives soft output.
- Includes the structure among symbols: pilot and header insertion and removal, OFDM carrier allocation.

**code**
- Encode takes data bits and gives channel bits. Decode takes **channel** bits (LLRs or hard) and gives **data** bits,
  LLRs when it can.
- Every transform is invertible: FEC, interleaving, rate matching, puncturing, scrambling, line codes such as NRZI, and
  attaching sync markers.
- Needs codeword boundaries to be known; finding them is sync.

**link**
- Decode takes aligned **data** bits and gives packets and messages: it splits at length fields or flags, checks integrity
  (CRC, FCS), strips headers and stuffing. Status (CRC result) goes in `ctx`.
- Encode builds the packets.

### Sync methods

**Block estimation and block-rate tracking are first class; per-sample loops are second class.**

- **First class, estimation:** estimate a parameter from a window of data with block operations: FFTs, correlations, least
  squares, matrix solves. Examples: the M-th power FFT for carrier frequency, square-law (Oerder-Meyr) timing plus
  interpolation, least squares on preambles and pilots, code-phase search by FFT correlation (`spread.acquire`), a block
  LS/MMSE equalizer. Existing pieces already work this way: `sync/phase/ff.py`, the `Acquire` estimators, `spread.acquire`,
  the DVB-S2 pilot phase fit.
- **First class, tracking:** a slow loop that runs **once per window** over the window estimates, carrying phase and rate
  from one window to the next and gating outliers. It is part of the estimator, not a separate stage.
- **Second class:** per-sample loops: Costas, PLL, Mueller & Muller, Gardner and TLL timing loops, LMS and CMA equalizers,
  decision-directed trackers. They stay available for signals and conditions where block estimation is not enough.

A profile's sync steps name **what** is estimated (`carrier`, `timing`, `frame`), not how. The receiver uses the first-class
method. When a signal needs a loop, the profile says so in `defaults`:

```json
"defaults": {"timing": {"method": "gardner", "alpha": 0.3}}
```

**Rules for feedforward estimators:**

1. **Windows sit at absolute stream positions,** not at block boundaries. A window covers stream samples
   `[k*W, k*W + L)` whatever blocks they arrived in, and history is carried between blocks, so the output does not depend on
   how the stream was cut. That is the same "any block length" contract every stage has today. It needs `Meta` to carry the
   absolute position of each block (see [Status](#status)).
2. **Parameters that drift are tracked across windows,** not held constant per window. A constant estimate per window is a
   staircase when the clock or carrier drifts (a free-running SDR oscillator is 10 to 20 ppm off and wanders). The
   block-rate tracker carries phase and rate from window to window, and the correction is evaluated per sample.
3. **Window length is set in time,** not in symbols. Oscillator phase wanders in time: `viterbirx` measured cycle slips
   with 8 ms blocks and none with about 2 ms, whatever the symbol rate. A window is as long as the noise needs, and no
   longer than the oscillators allow.
4. **Latency is declared.** Output waits for its window; the stage says how long in its estimates
   (`stream.est[name]["latency"]`, seconds). It is not added to `meta.delay`, which is the signal's group delay (what
   `meta.time_of` takes off): holding a window back does not move where a symbol was in time.

What there is (`rpps.sync`): `track` (the block-rate loop: a critically damped second-order loop over window estimates,
wrapped to the parameter's period, wide for the first windows while it pulls in, coasting through gated-out windows);
`time.OerderMeyr` (square-law timing, the symbol instants from the tracked line, cubic interpolation; follows a clock
200 ppm off); `phase.MPower` (M-th power of the phase alone: FFT acquisition of the frequency, then the residual phase
per window; the acquisition gate allows for the FFT's search, so noise is never acquired); `equalize.BlockLS` (least
squares per window against decisions, the taps carried from window to window). Profiles get them by default
(`timing: oerder_meyr`, `carrier: mpower`, `equalize: ls`); the loops are methods by name (`mm`, `gardner`, `costas`,
`lms`, `cma`).

### Profiles

A profile is a config file that defines one signal: which operators, in what order, with what parameters. It replaces
`chain`, which only covered FEC and needed its own special cases (Viterbi puncture-phase search, polarity, byte alignment);
those become ordinary sync and code steps.

```
config/profile/<family>/<mode>.json          config/profile/dvbs/qpsk_3_4.json, config/profile/ieee80211b/1m.json
rp.profile.load("ieee80211b/1m")             -> p.rx(), p.tx(): runners, `out = runner(block)`
                                                transmitting, `runner(block, fields=...)`: the fields of its frames
```

**One list of steps, in receive order.**
- A transform step names a transmit/receive pair. It runs as written on receive and inverted, in reverse order, on transmit.
- A sync step is receive-only and is skipped on transmit.
- So one file gives both directions, and sync sits exactly where it belongs.

Profiles take parameters like any definition (`params`, `{"param": ...}`, `{"expr": ...}`): a modulation and coding
choice, a satellite PRN, a cell ID. Each step references resources by name (see [Config resources](#config-resources)),
so a profile adds no new operator code. `defaults` holds method choices and loop parameters for sync steps that need them.

#### Example: IEEE 802.15.4, 2.4 GHz

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

What the loader builds:

```
rx = carrier sync + Stagger("i") + halfsine (matched) + timing sync + Spread(pair) decode
tx = Spread(pair) encode + Zeros(4) + halfsine + Stagger("q")
```

Transmit is the transforms in reverse, without the sync steps. The carrier is corrected before `Stagger("i")`, because a
phase offset mixes I into Q while Q is still half a symbol late.

#### Wave pairs

The profile loader knows how each wave step pairs:

| Step kind | Receive (as written) | Transmit (inverted) |
|---|---|---|
| `shape` with a **symbol** pulse (rrc, rc, halfsine, rect) | matched filter: the pulse time-reversed and conjugated (the same taps for symmetric real pulses) | `up.Zeros(sps)` + pulse |
| `shape` with a **frequency** pulse (gaussian, rect for FSK) | none: the discriminator in the `fm` step handles it | `up.Zeros(sps)` + pulse |
| `stagger` | `Stagger("i", delay)` | `Stagger("q", delay)` |
| `fm` | `FMDemod(fm_gain(h, sps, step))` + integrate over the symbol | `FM(h, step)` |
| `ofdm` | remove cyclic prefix + FFT | IFFT + cyclic prefix |

Whether a pulse is a symbol pulse or a frequency pulse is a property of the `pulse` resource (`"kind": "symbol"` or
`"frequency"`), not of the profile.

#### Header-driven control

Most standards decode a header that sets the format of what follows:

| Standard | Header | What it changes |
|---|---|---|
| DVB-S2 (VCM, ACM) | PLS, in every PLFRAME | modulation, LDPC rate, pilots, frame length, frame by frame |
| IEEE 802.11b | PLCP header, sent at 1 Mb/s | the payload's rate and modulation, mid-packet |
| IEEE 802.15.4, BLE, ADS-B | length or type field | packet length |
| P25 | NID (DUID) | frame type and length |
| Adaptive links | link-layer control message | code rate or modulation from a later point |

The receiver has to make the change at the right place and keep lock through it. So the architecture has a **control
path** that runs back up the chain:

1. **Fields are published, with a position.** A header step (code or link) publishes the fields it decoded, each with
   the absolute stream position from which it applies: the rest of this frame, the next frame, or a stated later frame.
   They go on the stream (`meta.stream`), like sync's estimates.
2. **The profile maps fields to parameters.** A `controls` section says which steps take which parameters from which
   fields, usually through a table resource. For DVB-S2, a `modcod` table maps the PLS value to a constellation, an LDPC
   rate, pilots on or off, and a frame length.
3. **Steps change parameters at a position.** Parameters are **fixed unless controlled**, and the declaration comes from
   where the change comes from: the header step's entry in `controls` names the parameters it drives, and only those can
   change. How a step takes the change is a property of the step: each parameter it can be controlled on is one of

   | Behaviour | What happens at the position | Typical parameters |
   |---|---|---|
   | **live** | changes in place; the step keeps its state | constellation, puncturing pattern, LDPC rate, frame length, the carrier estimator's power |
   | **restart** | the step is rebuilt from the position; what was in flight inside it is discarded and reported, as in the drop-and-report rule below | interleaver depth (its delay lines hold data interleaved the old way), Viterbi constraint length (its path metrics belong to the old trellis), FFT size, swapping a code for another |
   | **none** | cannot be controlled: the loader refuses a profile that tries | a sync tracker's own estimates, which a header should never overwrite |

   A **live** change keeps everything else going: the timing and carrier trackers keep their lock, and only what depends
   on the parameter changes (QPSK to BPSK changes the constellation and the carrier estimator's power, not the timing).
   Most header-driven changes are live at almost no cost, because the steps after frame sync already work frame by
   frame, so each frame simply uses its own parameters. A **restart** is what standards do anyway where state can't carry
   over: they define a boundary and start fresh there, and a decoder whose frames are terminated loses nothing.
4. **Positions translate between representations.** A position given in data bits is turned into the matching symbol and
   sample position through each step's rate and delay. This is why every block carries its absolute position (see
   [Status](#status)).
5. **A change can arrive late.** When a field applies to data that upstream steps have already processed (a link header
   that announces a change for its own payload, after the payload has gone through the decoder), the runner re-runs that
   span from a re-run buffer kept for the purpose. Standards that announce changes ahead (DVB-S2's header comes before
   its payload) never need it.
   - **The buffer is sized from the signal and adjusts with it.** It holds the longest span a header can reach back over:
     from the start of the data a field can apply to, to the point where that field is decoded. That span follows the
     current frame length and rates, so it grows and shrinks when a header changes them (a longer frame or a lower
     symbol rate needs more).
   - **Too late is dropped and reported.** A change whose data has already left the buffer is not applied backwards: the
     affected frames are dropped, and the drop is counted and reported on the stream (which change, which positions, how
     many frames), the way `dvbs2rx` records a refused frame as a gap. Processing continues with the change applied from
     the first position still in the buffer.

Frame-scoped steps are what make this practical: once frame sync has found a frame, the steps after it process frame by
frame, and each frame carries the parameters announced for it.

How it is done (`rpps.profile.control`, `rpps.profile.runner`):

- **Publishing.** A step calls `rp.profile.publish(meta, op, fields, at=..., out=..., span=...)`: `at` is the absolute
  position in its input from which the fields apply, `out` where that falls in what it gives in this call (`None`: none of
  its output is affected), `span` how many input items they hold for before the controls' defaults apply again (an 802.11b
  rate holds for its payload). A frame sync publishes `{"frame": k}` at each frame start. A step that drops or inserts
  items reports `rp.profile.anchor(meta, op, at, out)`, so positions carry through it exactly. What was published goes to
  `stream.control`: `log`, `fields(step)`, `reports` (late, dropped, restarted, no table row), `rerun`, `buffer`.
- **The controls section**, keyed by the publishing step's id (a step's `"id"`, else its kind):

  ```json
  "controls": {
    "frame": {"drive": {"pls": {"frame": "frame"}}},
    "pls": {"table": "dvbs2/modcod", "key": ["modcod", "short"],
            "default": {"modcod": {"param": "modcod"}, "short": 0, "pilots": {"param": "pilots"}},
            "frame": {"calc": "kbch - 80"},
            "drive": {"map": {"name": "constellation"}, "ldpc": {"name": "ldpc"}, "pilots": {"slots": "slots", "pilots": "pilots"},
                      "plscram": {"length": {"calc": "slots*90 + pilots*((slots - 1)//16)*36"}}}}}
  ```

  A source is a field, a column of the key's row, or `{"calc": ...}` over both, worked out when the fields arrive (an
  `{"expr": ...}` is worked out at load time, from the profile's parameters). The loader refuses a step or parameter that
  does not exist, an ambiguous id, and a parameter the step does not declare live or restart.
- **Steps** declare `CONTROLS` on their operators (`{"name": "live"}` on a constellation or a block code, `"restart"` on a
  convolutional code or a Forney interleaver) and take a live change through `control(**params)`. A change applied to a step
  is a frame start for it: frame-scoped steps (a header, pilot blocks, PL scrambling, a BBFRAME) restart their frame there.
  What the last frame left in a step (part of a codeword, when a timing slip cut that frame short) is dropped and
  reported, never joined to the next frame, and the step anchors the frame start, so positions after it stay exact.
  Each runner builds its own operators, since control changes them.
- **Ahead and late.** A change for a step after its publisher travels with the data like a tag: before a step runs, its
  input is split at the change's position and the change applied in between; after it ran, the position is carried to its
  output through the step's position map (anchors at its first run, at each change, and where it reported one). A change
  for a step before its publisher is carried back through the maps, and the runner re-runs from the earliest such step
  (the re-run span), which it runs in pieces with a checkpoint of their state at the start of each. The pieces are at least
  the `reach` the controls declare, carried into the span's units through the current rates, so a field never reaches
  further back than the last two checkpoints. Publications from one header share a `group`; a re-run keeps the earliest
  group and lets the re-run publish the rest again, since they came from data that was processed wrongly. Positions after
  a late change are carried back through what the change makes of the steps between (their maps are cut at the change),
  not through the pass that is being redone.
- **Transmit.** The runner publishes each frame's fields at the first step: the caller's (`fields=`, one dict or a list,
  one per frame), else the entry's `schedule` in turn, else its defaults; `frame` gives the items one frame takes at the
  first step (none: one frame per call). Those drive the steps up to the header step; a header step that publishes in turn
  drives the steps after it (the 802.11b PLCP builder knows where its payload starts, so the rate switches there).

On transmit the same fields are inputs, from two places:

- **A schedule in the profile**, such as cycling through MODCODs for a test signal, or a fixed sequence of packet types.
- **The caller, per frame**, for adaptive links or anything driven from outside.

A field the caller sets for a frame overrides the schedule for that frame. The profile builds the headers and switches the
steps to match.

#### Sources that start later in the chain

A source says which representation it produces and which sync it has applied. The profile runner skips every receive
step before that point:

- a radio or a capture gives samples: nothing is skipped;
- `gpu-demod`'s ring or `python-demod`'s symbol files give timed symbols at one sample per symbol with the carrier not
  removed: the signal, wave and timing steps are skipped, and carrier sync on symbols runs;
- a BBFRAME stream gives data bits: everything up to and including the code steps is skipped.

#### Parallelism

The goal is real-time processing, on CPU first and on a GPU or an FPGA later. Once frame sync has found a frame and its header
has set its parameters, everything after is per-frame work on a whole frame, with no state across frames, as `dvbs2rx` does.
A profile marks where that region starts (`"parallel": "<step id>"`); the loader checks that it holds (`rpps.profile.parallel`; internals in [Parallel runner](#parallel-runner)).

The unit is **batch**, not thread: what pays on every target is running N frames through one program.

```
Scheduler   frames -> batches of one key -> executor -> results in order
Executor    where a plan runs: inline, threads (a batch split by rows), processes, GPU, FPGA
BatchPlan   one per key (the parameters the header gave, the frame's shape): the region's kernels, shapes checked, `describe()`
Kernel      a pure function over [N, L] arrays, one frame per row, with its constants made once: `run(xp, batch)`
```

- An operator gives its kernel with `op.kernel(params, how)`; the operator-first API is unchanged. A step with no kernel runs
  frame by frame through its operators (`LoopKernel`), which is also the oracle every kernel is checked against. Regions can
  mix both.
- Kernels belong to generic operators (a constellation's demapper), never to a protocol. They use only what numpy and cupy
  share (`xp`), hold their constants (keystreams, permutations, a code's graph) so those can live on a device, and keep shapes
  fixed per key; a row that fails is marked, not raised.
- `BatchPlan.describe()` is data (kernel, lengths, constants): what a device kernel library or an HLS generator builds from. The
  JSON resources (LDPC address tables, BCH polynomials, interleavers) are already the constants.
- Per-sample loops stay serial and in the front (second class, above).

### Grey areas

Each is settled by the rule.

| Function | Category | Why |
|---|---|---|
| Matched filter | wave | prepares samples for sampling; does not decide when |
| Timing recovery (`MM`, `MAT`, `TLL`) | sync | estimates the sampling instant; the one sync step that changes representation |
| Differential coding on complex symbols (`Differential`) | symbol | symbols ↔ symbols |
| Differential coding on bits (NRZI) | code | bits ↔ bits |
| Manchester / biphase as a pulse (`ShapingBiphase`) | wave | symbols ↔ samples |
| Manchester as 0 → 01, 1 → 10 | code | bits ↔ bits |
| Spreading and despreading at one sample per chip (`Spread`) | symbol | chips are symbols |
| `Despread` at `sps` samples per chip, with a known code phase | symbol | the stage form of `Spread.decode` |
| Code-phase and Doppler search (`spread.acquire`) | sync | estimates timing and frequency |
| Correlating a known preamble in samples or symbols (DVB-S2 SOF, 802.15.4 SHR) | sync | finds the frame start |
| Searching for sync markers or sync bytes in channel bits (CCSDS ASM, DVB-S byte alignment) | sync | finds a codeword start |
| Attaching a sync marker on transmit | code | known structure among channel bits |
| Searching for HDLC flags in an unaligned data-bit stream | sync | finds a packet start |
| Resolving a phase ambiguity from the data (polarity, which rotation of the bit map) | sync | estimates an unknown |
| Inserting or removing pilots, physical headers, padding | symbol | known structure among symbols |
| Reading a physical header's fields (DVB-S2 PLS: modulation and code) | symbol | `symbol.Header` removes the header and decodes its short code (every word tried) into fields; the code and fields are a `header` definition |
| Equalizer | sync | estimates and removes the channel |
| OFDM IFFT/FFT and cyclic prefix | wave | symbols ↔ samples |
| OFDM carrier allocation and pilots | symbol | known structure among symbols |
| OFDM per-carrier pilot equalization | sync | estimates the channel |
| CRC guarding a packet | link | integrity of a packet |
| CRC inside FEC (LTE code-block CRC) | code, using the `crc` resource | resources are shared across categories |
| Analog demodulation: `FMDemod`, `AMDemod`, `SSBDemod` | wave | carrier modulation → baseband |
| Audio processing after it: de-emphasis, DC block | signal | samples → samples |
| Squelch | sync | detects that a signal is present |
| Channel impairments (AWGN, offsets, multipath) | signal (`signal.channel`) | samples → samples, signal agnostic |

**Soft information only flows down the receive chain.** The symbol transform gives LLRs, code transforms take and give
LLRs, the link transform takes hard bits plus their reliability. Sync on channel bits must pass LLRs through when it has them.

**Order matters inside sync.** A sync step belongs at the point where what it corrects can still be separated. Coarse
frequency usually comes before the matched filter, fine phase after timing, and a staggered signal's carrier before its
rails are realigned.

### Where things are

Each category is a package (`src/rpps/<package>`).

| Package | Category | What is in it |
|---|---|---|
| `source` | source | sources and sinks of blocks: files and their formats, sockets, radios through pysdrlib |
| `signal` | signal | filters (`signal.filters`: FIR, IIR, windows, pulse shapes, fractional delay, polyphase channelizer), resampling (`signal.sample`), `channelize`, audio conditioning (`DCBlock`, `Deemphasis`), and `signal.channel`, the simulated medium |
| `sync` | sync | first class: `time.OerderMeyr`, `phase.MPower`, `equalize.BlockLS`, all tracked by `track`; frame sync (`FrameSync`, `Delimiter`), pilots (`PilotPhase`, `PilotEqualizer`), alignment on bits (`Align`, `Puncture`), `Acquire`, `acquire_code`, `Squelch`. Second class, by name: `time.MM` (and Gardner), `Costas`, `AGC`, `PLL`, `equalize.LMS`, `equalize.CMA` |
| `wave` | wave | `Stagger`, `FM`/`FMDemod`, `OFDMTx`/`OFDMRx`, `AMDemod`, `SSBDemod`; pulse shapes are `signal.filters` operators on the `pulse` resource |
| `symbol` | symbol | `Constellation` (and the PSK, QAM, APSK, ASK generators), `Differential`, `Spread`/`Despread`, `Carriers`, and frame structure: `Header`, `Pilots`, `PhaseScramble` |
| `code` | code | FEC, interleaving, rate matching (`code.types`), scramblers (`code.scram`), `Uncoded` |
| `link` | link | CRCs, HDLC framing, `Header` (fields, CRC, payload length) |
| `profile` | profile | loading (`profile`), the steps (`steps`), header-driven control (`control`), runners (`runner`) |
| `sequence` | (a resource engine) | recurrences and Gold sequences, chip codes, LFSR registers: what scramblers, spreaders and searches share |
| `analysis` | (outside the data path) | spectra, rate estimation, survey, blind analysis of modulation and FEC; can write definitions |
| `base`, `math`, `utils`, `viz` | (cross-cutting) | `Data`, `Stage`, `Pipe`, `Stream` and the type tags; finite fields and GF(2); helpers; plots |

### Config resources

Configs are organized by **the kind of thing they define**, not by the package that loads them. An operator in any category
references a resource by name. The loader picks the implementation from the file's `base` and `type`, so a new
implementation of a function is a new loader, and a new instance of it is a new file.

| Resource | Where | Used by |
|---|---|---|
| `constellation` | `config/constellation` | symbol |
| `pulse` | `config/pulse` | wave (shaping, matched filters). Each pulse says `"kind": "symbol"` or `"frequency"` |
| `filter` | `config/filter/{fir,iir,window}` | signal |
| `sequence` | `config/sequence/{recurrence,chips,set,gold}`: LFSRs and Gold sequences (LTE, DVB-S2 PL, P25, 802.11a, BLE, the LFSRs of CCSDS and DVB), chip codes (Barker, 802.15.4, GPS C/A) | code (scramble), symbol (spread, PL scramble), sync (code-phase search). Sync markers and start-of-frame words are still in the steps that use them |
| `scramble` | `config/scramble/{additive,feedthrough}` | code: how a sequence is applied, additive (naming a sequence, with a frame or period restart) or self-synchronizing (its own register, since the data runs through it) |
| `fec` | `config/fec/{block,conv}` | code |
| `interleave` | `config/interleave/{block,forney}` | code |
| `crc` | `config/crc` | link, and code operators that carry one |
| `frame` | `config/frame` | link: frame formats such as HDLC |
| `header` | `config/header/physical` (known symbols and a short code carrying fields), `config/header/link` (fields, CRC, payload length) | symbol, link, and sync (a frame sync correlates a header's preamble, a delimiter search a link header's word) |
| `table` | `config/table` | profile controls: rows of parameters looked up by header fields (DVB-S2 MODCODs, 802.11b rates) |
| `profile` | `config/profile` | whole signals |

Example: `scramble/additive/lte_pdsch.json` is a scrambler that names the LTE Gold sequence and works out its seed:

```json
{"base": "scramble", "type": "additive", "params": {"pci": ..., "ns": ..., "rnti": ..., "q": ...},
 "sequence": {"name": "lte_gold", "args": {"c_init": {"expr": "(rnti << 14) + (q << 13) + ((ns//2) << 9) + pci"}}}}
```

The same `lte_gold` sequence (`rp.sequence.load("lte_gold", c_init=...)`) is available to any other operator. A spreading code, a scrambler, a preamble detector
and a sync-marker search share one sequence engine.

### Type tags

Every block of data carries two tags:

- **Form**, `dtype` in `base/types.py`: SAMPLES, REAL, SYMBOLS, LLR, BITS, BYTES, MSG. How the values are stored.
- **Representation**, new: SAMPLES, SYMBOLS, CHANNEL, DATA, PACKETS. Where in the chain the data is.

Two tags because the forms cross representations: LLRs exist both before the decoder (channel bits) and after it (data bits,
from a soft-output LDPC or turbo decoder), and so do hard bits.

`pproc` (SERIAL, FILTER, SYNC, MOD, MAP, CODING, SCRAM) records the last operation. It becomes a `Category` enum: SOURCE,
SIGNAL, SYNC, WAVE, SYMBOL, CODE, LINK.

| Category | Accepts | Gives |
|---|---|---|
| signal | SAMPLES (form SAMPLES or REAL) | the same |
| sync | any representation | the same representation; timing recovery gives SYMBOLS from SAMPLES |
| wave | SYMBOLS (TX), SAMPLES (RX) | SAMPLES (TX), SAMPLES (RX, for the sampler) |
| symbol | CHANNEL (encode); SYMBOLS (decode) | SYMBOLS; SYMBOLS or CHANNEL (form LLR) |
| code | DATA or CHANNEL (encode); CHANNEL or DATA, form LLR or BITS (decode) | CHANNEL; DATA, form LLR or BITS |
| link | PACKETS (encode); DATA (decode) | DATA; PACKETS (form BYTES or MSG) |

Each operator declares what it takes (`takes(encode)`), and the operators check it before doing any work, so a link decoder
refuses channel bits and a constellation refuses samples.

- **A code transform is often several steps** (Viterbi, deinterleave, Reed-Solomon, descramble), so each code step takes
  both channel and data bits; the boundary that is enforced is the link's.
- **An uncoded link says so:** `rp.coding.Uncoded()` is the code step of a link with no FEC (ADS-B, many packet radios).
  It turns channel bits into data bits, LLRs staying LLRs.
- **A complex array is samples unless it says otherwise:** `rp.Data(x)` is SAMPLES, `rp.Data(x, sps=1)` is SYMBOLS.
- **Stages take samples and symbols;** `BitStage` takes channel and data bits and gives the same form back.

### Status

Everything above is implemented, apart from the items listed under Not yet. How the code is built (streaming rules, native
kernels, the parallel runner's internals, how to add things) is in the sections that follow.

| Area | State |
|---|---|
| Contracts | Every operator declares what it takes (`Rep`, `Category`) and checks it before working (`tests/unit/test_contracts.py`) |
| Stream position | `meta.pos` is the absolute position of a block's first item at its own rate; `meta.time_of(k)` gives source time |
| Streaming on bits | `BitStage` and `sync.Align` (bit offset, pattern phase, polarity, tracking) |
| Packages | One per category (see [Where things are](#where-things-are)); no aliases for the old names |
| Mixed modules split | `AMDemod`/`SSBDemod` in `wave`, `Squelch` in `sync`, `DCBlock`/`Deemphasis` in `signal`, spreading in `symbol`, code-phase search in `sync`, OFDM in three operators (`symbol.Carriers`, `wave.OFDMTx`/`OFDMRx`, `sync.PilotEqualizer`) |
| Profiles | `ieee802154/2450`, `ieee80211b/{1m,2m,long}`, `dvbs/qpsk_3_4`, `dvbs2/normal`; `chain` is gone, its special cases are ordinary sync and code steps |
| Header-driven control | `rpps.profile.control` and `rpps.profile.runner`: publishing, tables, live and restart changes, late changes re-run from checkpoints, drops reported |
| First-class sync | `sync.time.OerderMeyr`, `sync.phase.MPower`, `sync.equalize.BlockLS`, tracked by `sync.track`; the profile defaults for `timing`, `carrier`, `equalize` |
| Config by resource | `config/<resource>/`; `sequence` split from `scramble`; `rp.spread` became `rp.sequence` |
| Parallel runner | Frame-parallel region after the frame boundary: `ParallelRunner`, `Scheduler`, thread and process executors, batch kernels, C++ demapper and front-end kernels. Output equals the serial runner's bit for bit |

**Not yet**
- `dvbs2/normal`: short frames, 16/32APSK and dummy frames (their PLS is reported, the frame is not decoded).
- A first-class carrier estimator for staggered signals (802.15.4 still needs its carrier taken off first), and fractionally
  spaced block equalization.
- Sync markers and start-of-frame words as sequences (the DVB-S sync bytes, the 802.11b SFD and the DVB-S2 SOF are written in
  the steps and headers that use them).
- The parallel runner numbers its output from 0 and a dropped frame leaves no gap (the serial runner keeps positions).
- Fusing the matched filter with Oerder & Meyr timing (its square-law line needs the filter at every sample), and evaluating
  the pulse exactly at fractional instants instead of interpolating.
- A GPU backend (`xp = cupy`) and an FPGA backend from `BatchPlan.describe()`. The seams are in; what is missing is kernels
  for the front's estimators and device-resident constants in `Kernel`.

## Core abstractions

**Data** (`base/data.py`). `rp.Data(x)` wraps an array with its `Meta`. A complex array is samples; `rp.Data(x, sps=1)` is
symbols; bytes are bits. It carries the sample rate (`Fs`), `sps`, centre frequency, group delay, the stream state and the
absolute position of its first item (`meta.pos`). Because the data carries all of that, nothing needs setting up or resetting.

**Operators** do everything:

| Operator | Means | On |
|---|---|---|
| `data * pipe` | encode, modulate, scramble | transforms (`Pipe`) |
| `data / pipe` | the inverse | transforms |
| `data + stage` | process a signal | `Stage` |
| `pipe * pipe`, `stage + stage` | a reusable `Pipeline` | either |
| `data >> sink` | write out | `Sink` |

**Pipe, Stage, BitStage** (`base/rpps.py`).
- `Pipe` is an invertible transform: `encode(data)` and `decode(data)`.
- `Stage` is a DSP stage on a stream of blocks. Settings are constructor arguments; whatever carries from one block to the next
  lives in a state object from `new_state(meta)`, kept in the stream, so the stage object stays reusable. `step(samples, st,
  meta)` processes a block and sets `meta.op` (the output rates).
- `BitStage` is the same on channel or data bits (LLRs positive for 0, or bools), keeping their form.

**Type tags** (`base/types.py`): every block has a *form* (`dtype`) and a *representation* (`Rep`), and an operator checks what
it takes before working (see [Type tags](#type-tags)). `CATEGORY` records which category it is. `SCOPE` is `STREAM` (default:
state crosses frames) or `FRAME` (nothing held across a frame start), which the parallel runner needs.

**Stream and Meta** (`base/meta.py`). `Stream` holds the per-stage states, estimates (`stream.est[name]`: frequency offset,
timing, lock, latency), detections (`det`), control publications (`stream.control`) and load statistics (`stream.load()`).
Estimates go on the stream, never into the data.

## Streaming rules

Every stage and transform must behave the same however the stream is cut. The tests check it (`test_stream.py`,
`test_position.py`, per-stage block-cut tests).

1. **Any block length, including 0.** Never drop or repeat samples across blocks.
2. **State lives in the stream state**, not on the object, so one stage can serve several streams.
3. **Whole blocks:** an operator that works on `n` items at a time uses `rp.base.rpps.align(ctx, key, x, block, pad)`: the
   leftover waits in `ctx` for the next call, and only a flush pads it (encoding) or drops and reports it (decoding).
4. **Positions are absolute.** A stage counts its outputs from where its input was when it first ran. A stage that drops or
   inserts items reports it (`rp.profile.anchor`), so positions after it stay exact.
5. **Feedforward estimators work on windows at absolute stream positions**, track drifting parameters across windows, set the
   window in time not symbols, and declare latency in the estimates (not in `meta.delay`).
   See [Sync methods](#sync-methods).
6. **Soft information flows down the receive chain.** Sync on channel bits passes LLRs through.

## Definitions (config)

Codes, scramblers, constellations, pulses, filters, sequences, headers and profiles are JSON files, not code. A new instance
of something that exists is a new file; a new kind needs a loader.

- Layout: `config/<resource>/[<group>/]<name>.json`. The file's `base` picks the loader within the resource, its `type` the
  implementation. The resource table, naming rules and file header are in [config/README.md](config/README.md).
- Search order: directories added with `rp.add_config_path(path)` or in `$RPPS_CONFIG` first, then the shipped `config/`.
  In a checkout that is `config/` next to `src/`; an installed package carries a copy (`rpps/config`), put there by
  `setup.py` at build time. So a wheel that lacks `config/` is broken: CI checks the count.
- Parameters: a definition can declare `params`, and any value can be `{"param": ...}` or `{"expr": ...}` (evaluated at
  load time). A step's resource arguments go in `"args"`.
- Loaders register per resource (`_config.Domain.register(category, loader)`); `ls`, `find` and `load` then work on it.
  The keys each loader accepts are documented in the header of its module (linked from [config/README.md](config/README.md)).

## Native kernels

`rpps._native` (C++17, pybind11, `csrc/rpps_native/*.cpp`) holds the per-sample loops and decoders: AGC, Costas, timing,
equalizers, IIR, Viterbi, trellis, BCJR/turbo, LDPC, Reed-Solomon, BCH, soft demapping, resampling, FIR decimation, the
carrier estimate and the frame sync. They are optional: without pybind11 or a compiler the build warns and RPPS runs its
Python loops.

**The Python loop is the reference.** Every kernel has the pure Python version in the stage that calls it, and the tests
compare the two (`test_native*.py`, `test_parallel.py`, `test_batch_kernels.py`). A kernel changes speed, never results
beyond rounding.

| Variable | Effect |
|---|---|
| `RPPS_NO_NATIVE=1` | never load the kernels |
| `RPPS_REQUIRE_NATIVE=1` | a failed native build is an error (CI and wheel builds set it) |
| `RPPS_THREADS=n` | threads for kernels that spread work over cores |
| `RPPS_CONFIG` | extra config directories (path-separated) |

Stages look `rpps._accel.native` up when they run, so tests can switch it off mid-session.

Adding a kernel:
1. Write it in `csrc/rpps_native/<name>.cpp` with a `register_<name>(m)` function, declare it in `common.hpp`, call it from
   `module.cpp`. `setup.py` globs `*.cpp`, so there is nothing to add to the build.
2. Keep the Python loop in the calling stage as the fallback and reference; guard the call with `_accel.native`.
3. Release the GIL for anything long, so threads (`.pipelined`, the parallel runner) run it in parallel.
4. Add a test that compares the kernel to the Python loop on random data, with and without `RPPS_NO_NATIVE`.

Pitfalls found the hard way:
- **Use `long long` for loop and tap indices.** CPython's build flags include `-fno-strict-overflow`, so `int` indices are not
  widened and a kernel that was fast alone ran 3x slower inside the module.
- **`target_clones("avx2,fma")` makes separate clones**, not one with both; use `"arch=haswell"` for both.
- **Keep BLAS products out of per-block paths.** OpenBLAS threads inside a call and spins between calls, which slows the
  single-threaded numpy and native code around it (block mode was 20-60% slower until the frame sync's product became one C++
  call). `OPENBLAS_NUM_THREADS=1` shows whether it is the cause.
- **Benchmark best-of-N in one process.** A shared machine's load makes runs across processes incomparable.
  `tests/bench_stages.py`, `tests/parallel_bench.py` and `tests/resample_bench.py` do this.

## Parallel runner

Real-time use needs more than one core. Once frame sync has found a frame and its header has set the parameters, the rest is
per-frame work with no state across frames. The design is in [Parallelism](#parallelism); in code:

```
profile.rx(parallel=rp.parallel.Parallel("thread", workers=4))     -> ParallelRunner
  front   the ordinary Runner, stopped at the region (Runner.stop, _finish)
  cut     the changes the front resolved split the region's input at each frame start
  Scheduler   frames -> batches of one key -> Executor -> results in order
  BatchPlan   per key: the region's kernels, shapes checked, describe()
  Kernel      a pure function over [N, L] arrays, run(xp, batch); LoopKernel runs frame by frame through the operators (the oracle)
```

- The loader refuses a region with stream-scoped steps, a publishing header or a late change (`rpps.profile.parallel`);
  mark where it starts with `"parallel": "<step id>"`.
- An operator gives its kernel with `op.kernel(params, how)`. Kernels belong to generic operators (a constellation's demapper),
  never a protocol. They use only what numpy and cupy share (`xp`), hold their constants, and keep shapes fixed per key; a
  row that fails is marked, not raised.
- Kernels so far: `Demap`, `BlockDecode` (block codes and block interleavers), `Rotate`, `Gather`, `XorMask`, `PilotLine`.
  Each is checked row for row against the operator run frame by frame (`tests/unit/test_batch_kernels.py`).
- Executors: inline, threads (a batch split by rows), processes (`start_method` is `fork` where there is one, else `spawn`,
  so the script needs a `__main__` guard). Each process builds its own region from a picklable recipe (`RegionSteps`).
- On `dvbs2/normal` the output equals the serial runner's bit for bit, for any block cut, worker count and batch size.
- One demapper: `rpps.math.demap` for rows of symbols; `Constellation.llr` and `Demap` both use it, so a frame comes out the
  same alone or in a batch. It runs in C++ (`demap_llr`) with the array version as fallback and reference.
- Front-end speedups already in: FFT overlap-save for FIR and correlation (`signal.filters.convolve`), a phasor recurrence for
  the carrier line (`sync.track.line_phasor`), native carrier windows and frame sync, and a matched filter fused into the
  timing step when timing is known (`sync={"timing": {"fuse": False}}` turns it off).
- Measure with `python tests/parallel_bench.py 64 4` (64 synthetic frames, 4 workers).

## Testing

```
python -m pytest                       # tests/unit, everything
python -m pytest -m "not slow"         # skip the chain-decoding tests
RPPS_NO_NATIVE=1 python -m pytest -m "not slow"     # the Python loops
python -m pytest tests/unit/test_remez.py tests/unit/test_iir.py     # one file
```

- `tests/unit/conftest.py` sets the Agg matplotlib backend and gives `rng` (seeded), `bits` and `awgn` helpers.
- **RPPS must not need scipy.** The suite runs without it; the oracle tests (remez, resample_poly, convolve, the IIR designs,
  channel sync) compare against scipy and are skipped when it is not installed. Install `.[oracle]` to run them.
- `test_contracts.py` runs every definition against its category's contract. New definitions are covered automatically.
- Native code is tested against its Python reference (above). Stream behaviour is tested by cutting the same input into
  different block sizes and expecting the same output.
- `projects/tests` hold the tests of the worked examples; they need no radio.
- CI runs the fast tests on two Pythons and two operating systems, then the Python-loop run, the oracle tests and the slow
  tests on Python 3.12 (see [CI and releases](#ci-and-releases)).

## Adding things

**A constellation, code, scrambler, pulse, CRC, sequence, header or profile:** write the JSON file in the right
`config/<resource>/` folder with `base`, `type`, `comment` and, where there is one, `ref`. Run `python -m rpps ls <resource>`
to see it, `python -m rpps fmt --check` to check the style, and the contract tests cover it.

**A new implementation of a function (a new code type, a new sync method):**
1. Put it in the package of its category (the rule: what it takes and gives, see [The rule](#the-rule)).
2. Subclass `Pipe` (invertible), `Stage` or `BitStage`; set `CATEGORY`, `takes`, and `SCOPE` if it holds nothing across frames.
3. Register its loader so a file's `type` can select it.
4. Write the Python version first, with the block-cut test; move the hot loop to C++ after, keeping the Python as the reference.

**A sync estimator:** estimate from a window with block operations (FFT, correlation, least squares), at absolute positions,
tracked across windows by `sync.track`. A per-sample loop is second class and selected by name in the profile's `defaults`.

**A new signal:** a profile in `config/profile/<family>/<mode>.json`: one list of steps in receive order, with `params`,
`defaults` for sync methods and, if a header changes later steps, a `controls` section and a `table`. It should need no new
operator code. See [Profiles](#profiles) for the format and the 802.15.4 example.

**A source:** subclass `Source` in `rpps.source`, say which representation it produces and which sync it has applied, set the
rates it knows, and give each block its absolute position. The profile runner skips the steps before that point.

## CI and releases

Workflows are in `.github/workflows/`:

| Workflow | When | Does |
|---|---|---|
| `tests.yml` | push to `main`/`dev`, pull requests | tests on Python 3.12 and 3.13, Ubuntu and Windows; the Python-loop run, oracle and slow tests on 3.12; builds the sdist and wheel and runs the examples from a clean environment; builds the docs |
| `deploy-docs.yml` | push to `main` | builds the docs (pdoc) and deploys them to GitHub Pages |
| `python-publish.yml` | a GitHub release is published | builds wheels (cibuildwheel: Linux x86_64 and arm64, Windows) and the sdist, checks the tag matches the version, publishes to PyPI by trusted publishing |

To release:
1. `python builder.py -v p` (or `m`, `M`) to preview the bump, then `-r` to do it. It edits `pyproject.toml` and
   `src/rpps/__init__.py` and commits; it does not tag.
2. Merge to `main`.
3. Publish a GitHub release tagged `v<version>` (for example `v0.3.1`). The publish job refuses a tag that is not exactly
   `v` plus the version in `pyproject.toml`. Only `v#.#.#` tags are used.

## Conventions

- **Operators first.** The API is `*`, `/`, `+`, `>>`; keep new features inside it rather than adding setup calls.
- **No aliases.** When something is renamed or moved, drop the old name and fix the callers.
- **Be explicit.** Fix a caller rather than loosen a contract check.
- **A function belongs to the category named by what it takes and gives**, not what a standard calls it
  ([the rule](#the-rule)).
- **Read the standard** when a decode does not work, instead of guessing from memory.
- **Examples live in `projects/`**, analysis in `scripts/` or `tests/dev`. Reuse RPPS components there so the useful parts
  can move into the package.
