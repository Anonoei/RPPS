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
- Your own operator is a step by module path, in the category it does: `{"signal": "class", "class": "mypkg.Gain",
  "args": {...}}`. A `signal` or `sync` class is a `Stage` run on receive only; a `wave` class a `Stage` on receive
  and `"tx_class"` the one on transmit; a `symbol`, `code` or `link` class a `Pipe`, decoding on receive and encoding on
  transmit. A class whose `CATEGORY` is another category's is refused.
- Variants of one signal (QPSK or 16QAM, RRC 0.2 or 0.35, a step only some modes have) are params with `"choices"` and
  steps with `"when"`: [Definitions](definitions.md#variants-choices-and-when).
- `"controls"` lets a header's fields drive other steps (DVB-S2's MODCOD, 802.11b's rate): see
  [`rpps.profile.control`](../api/profile.md).
- Bursts, preambles, branches, resource grids and the parallel region are each a step or a key:
  [`rpps.profile`](../api/profile.md) lists them and where each is documented.

A new signal should need no new operator code. `examples/own_definitions.py` ([Definitions](definitions.md)) writes one
from scratch in a directory of your own.

### A unique word before each frame

A word the modem sends ahead of each block of coded bits (a sync marker, a unique word) is a `sequence` of type word,
of any width: 100 bytes are 200 hex digits.

```json
{"base": "sequence", "type": "word", "word": "0x1acffc1d...", "bits": 800, "order": "msb"}
```

There are two places to find it, both after demodulation and before the FEC:

- **Among bits:** `{"sync": "delimit", "sequence": "<word>", "length": <bits after it>, "keep": true, "polarity": true,
  "errors": 40}` after the symbol mapping, then `{"code": "marker", "sequence": "<word>", "length": ...}` to strip it.
  `errors` is how many of the word's bits may be wrong (`{"param": ...}` works too), and `polarity` matches the inverted
  word too. Use this when the carrier leaves no ambiguity, or only an inversion (BPSK).
- **Among symbols:** `{"sync": "frame", "sequence": "<word>", "threshold": 0.3}` before the symbol mapping. The profile
  maps the word to symbols through its own steps, and the match measures the word's phase, so a QPSK or 8PSK carrier
  left a quarter turn off is put right by the word. Its tolerance is the normalised correlation `threshold`: random
  data correlates about 1/sqrt(symbols), 0.05 for a 400-symbol word.

How many errors to allow: as many as the word's own bit errors need, but no more than random bits could match.
`rp.sync.word_errors` gives the second bound: the most errors a match may have while random bits stay under a
false-alarm rate over the places searched. A long word can tolerate a lot:

```python
import rpps as rp
assert rp.sync.word_errors(800, 1e-9) == 314                  # 100 bytes: 314 wrong bits still can't be chance
assert rp.sync.word_errors(32, 1e-3, positions=8160) == 1     # a 32-bit ASM searched over a CCSDS frame
```

For an unknown signal, `rp.analysis.framing.payload.fixed_word(bits)` finds such a word blind among demodulated bits
(bit errors and all). It gives the word, its period, and the bits each frame's copy differs by, with an `errors` to
start from. `rp.analysis.framing.frames.find(symbols)` does the same among symbols.

### A preamble that says what follows

A frame sync can look for several words at once and say which it found. Each alternative is named, and the one that
matched is published with the frame as `word`, with its own fields, so `controls` drive the steps after it from a table,
the same way a header's fields do:

- `"sequences": {"bs_voice": "dmr_bs_voice", ...}`: words given as bits, mapped to symbols by the profile's own steps
  (DMR's eight sync patterns: voice or data, base station or mobile).
- `"zc": {"roots": [25, 29, 34], "length": 63, "shifts": [0]}`: a set of Zadoff-Chu sequences, published with `root`
  and `shift`. One complex sequence alone is a definition of type `zc` named by `"sequence"`.
- `"before": 54, "length": 132`: the frame starts that many symbols ahead of its word (a sync word in the middle of a
  burst) and is that long; `length` can be driven by a control too.
- Real levels (an FM discriminator's) stay real, and the match's phase is a polarity. `"scale": true` puts the levels
  back where the word's are; `"polarity": false` matches in phase only, for a discriminator whose sense is known, and
  is needed when alternatives are each other inverted (DMR's voice and data patterns).
- Every frame also publishes what its preamble measured: `scale`, `phase`, `cfo` (cycles a symbol, from the phase between
  the word's halves), `esn0` (the frame's symbols over the noise left after the word). `"cfo": true` takes that
  frequency off the frame; `"segments": 4` correlates the word in parts so a carrier that turns across a long word
  doesn't hide it. Offsets past about 1/L cycles a symbol are for a carrier step before.
- `"classify": ["psk/qpsk", "psk/8psk"]`: when the word does not say the modulation, each frame's symbols are scored
  against the candidates and the best published as `modulation`.

On transmit, `{"symbol": "marker", "zc": {"length": 63, "root": 25}}` puts known symbols before each frame (its `root`,
`shift` or `sequence` set per frame by a control), and strips them on receive. `{"symbol": "puncture", "every": 36,
"at": [35]}` drops a symbol at those places of every period (P25's status symbols) and puts `value` there on transmit.

```python
--8<-- "examples/zc_modes.py"
```

One frame can hold several codes: a branch step can give each branch only its `take` (item ranges of every frame,
joined) and only the frames its `select` expression accepts, over the fields a trunk step published for that frame.
Frames are a fixed `frame` length, or start where a trunk control drives the branch step's `frame` (a frame sync's
`{"tsdu": {"frame": "frame"}}`) and run to the next. A range's end can be null (to the frame's end), and
`{"ranges": [[64, null]], "unit": 196}` keeps whole groups only: `profile/p25/tsdu` reads one to three TSBKs a frame
that way. `profile/dmr/burst` reads every burst's slot type (Golay (20,8), split around the sync word) and only the
data bursts' BPTC payload:

```json
{"branch": "burst", "frame": 264,
 "take": {"slot": [[98, 108], [156, 166]], "data": [[0, 98], [166, 264]]},
 "select": {"data": {"step": "fs", "when": "word == 'bs_data' or word == 'ms_data'"}},
 "branches": {"slot": [...], "data": [...]}}
```

## Burst time plans

A TDMA or MF-TDMA network's terminals send bursts in timeslots of a repeating plan: superframes of frames, each frame a
grid of carriers and timeslots, each slot a burst type, and which terminal has which slot. A burst step that knows the
carriers cuts each burst on its own carrier, and with a plan names its place:

- `{"sync": "burst", "carriers": [{"freq", "width", "rate"}, ...], "sps": 4}`: bursts on many carriers of one wideband
  capture, each carrier brought to baseband at `sps` samples a symbol (`rp.sync.Bursts`). Each burst's fields
  (`carrier`, `freq`, `start`) reach the steps after it; its type comes from its unique word (a frame sync with
  `sequences`), its length and code from a table keyed on the word.
- `"plan": "<name>"` (or `control(plan=...)`): each burst is placed in the plan (`superframe`, `frame`, `slot`,
  `tx_type`, `assignment`), and its decoded messages carry those under `fields["burst"]`. The plan's time in the capture
  (its origin) is given, or found: the bursts vote for it, and their unique words keep it there (the clock is tracked).
  `"gate": "plan"` cuts every slot the plan gives, whether its energy shows or not.
- Transmitting, the plan places each burst: `tx(data, fields={"superframe", "frame", "slot"})`.

A plan (`rpps.plan`) comes from where the network keeps it:

- **Over the air.** DVB-RCS2's hub broadcasts it in its forward link: SCT, FCT2, BCT and TBTP2 tables in GSE packets,
  and the network clock (NCR). `profile/dvbrcs2/forward` reads them and `rp.plan.builder("dvbrcs2/forward")` makes the
  plan and the clock of its messages. The builder is a definition too (`config/plan/dvbrcs2/forward.json`): which
  tables, how their fields map onto the plan model both ways, and how the clock references pair with frame starts.
- **A definition**, `config/plan/<family>/<name>.json`, in the same model: a standard's fixed structure, a test. A
  sequence's `hop` moves its frames from channel to channel (an expression, or an algorithm by name: `ble_csa1`); a
  frame's `repeat` and `every` make a train of bursts.
- **Learned from the bursts**, when the forward link is not heard: `rp.analysis.spectrum.burst.learn_plan(bursts)` finds
  the BTU, the frame and the slots, and saves a definition (`.save(name)`).

Whether a plan, from anywhere, explains the bursts: `rp.analysis.spectrum.burst.check_plan(plan, bursts)` fits its origin
and clock to them and puts each in a slot of its carrier (in a slot of its kind, another kind's, overrunning it, or
in none), and `rp.viz.bursts(None, check, fold=True)` draws a raster of superframes, one a row: a right plan keeps the
bursts inside its slots' columns, a wrong frame slants them, a wrong BTU puts them across the columns' edges.
`rp.viz.burst_timing` draws each burst's start error against time, and `rp.viz.plan_evidence` what learn_plan chose its
BTU and frame on (or, from the `NoPlan` it raises, why every period was passed over).

```python
--8<-- "examples/mftdma.py"
```

## Decoded messages, and only some of them

Every frame a profile receives is also a record, `rp.profile.Message`: its kind, the fields it decoded, its payload,
`crc_ok`, and where it starts in the stream (`sample`, `time`, `freq`). What a payload says goes in
`fields["payload"]` when its header or frame definition has `"payload_fields"` (AX.25 addresses, BLE AD structures),
and a step's `"message"` fields are a schema too (DVB-S's TS adaptation field and PCR): see
[Definitions](definitions.md#fields).

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

They also narrow a blind search. `rp.profile.fits(name, baud=..., bw=..., Fs=..., seconds=..., input=...)` says why a
signal measured so can't be a profile, and `rp.profile.candidates(...)` lists the variants that may be. A profile whose
rate is set elsewhere, whose channel is narrower than the signal, or which takes audio, is never built:

```python
import rpps as rp
print(rp.profile.fits("ble/adv", baud=250e3))                     # its rate is 1e+06, not 250000
print(sorted({n for n, _ in rp.profile.candidates(baud=27.5e6)})) # the open-rate profiles: CCSDS, DVB-S, DVB-S2
```

## More than one core

A profile marks where its frame-parallel region starts; `p.rx(parallel=rp.parallel.Parallel(kind, workers=n))` runs it
in batches ([`rpps.parallel`](../api/parallel.md)).

```python
--8<-- "examples/parallel_rx.py"
```
