# Definitions

Codes, scramblers, constellations, pulses, filters, sequences, headers, tables, profiles and systems are JSON, not code: a new
instance is a new file, a new kind is a new loader. Definitions are organized by **resource** (what they define, not
which package loads them), one folder each: `config/<resource>/[<group>/]<name>.json`. A file's `base` picks the loader
within the resource and its `type` the implementation; the keys each accepts are in the docstring of the module linked.

- **Search order:** `rp.add_config_path(dir)` and `$RPPS_CONFIG`, then the shipped `config/` (`rpps/config` in a wheel).
  Your directory is searched first, so it can also override shipped definitions.
- **Loading:** `rp.load("<resource>/<name>", **params)` loads any definition; each package has its own loader too
  (table below). `params` with `{"param": ...}` and `{"expr": ...}` values are evaluated at load.
- **Listing:** `python -m rpps ls [<resource> ...]`, `python -m rpps show <resource>/<ref>`, `rp.tagged("dvb-s2")`.
- **No aliases:** one name per definition, one reference per loader.

```
python -m rpps ls                     # everything shipped
python -m rpps ls sequence scramble   # some resources
```

## Your own signal, in your own directory

```python
--8<-- "examples/own_definitions.py"
```

## Resources

| Resource | Holds | Example | Load | Keys |
|---|---|---|---|---|
| `constellation` | point sets (or PSK/APSK rings) and their bit maps; shapes by name (16QAM) come from `rp.symbol.generate` | `psk/qpsk` | `rp.symbol.load("psk/qpsk")` | [symbol/constellation.py](https://github.com/Anonoei/RPPS/blob/main/src/rpps/symbol/constellation.py) |
| `pulse` | symbol and frequency pulses (`"kind"`): `rrc`, `rc`, `halfsine`, `gaussian`, `rect/symbol`, `rect/frequency` | `rrc/0p35` | `rp.signal.filters.load("pulse/rrc/0p35")` | [signal/filters/serial.py](https://github.com/Anonoei/RPPS/blob/main/src/rpps/signal/filters/serial.py) |
| `filter` | FIR, IIR, windows | `fir/windowed/lowpass_0p1_60db` | `rp.signal.filters.load("fir/windowed/lowpass_0p1_60db")` | [signal/filters/serial.py](https://github.com/Anonoei/RPPS/blob/main/src/rpps/signal/filters/serial.py) |
| `sequence` | bit sequences (recurrences, Gold), chip codes (Barker, PN sets, GPS C/A) fixed words (sync markers, access addresses: `word/ccsds_asm`) and Zadoff-Chu sequences (`zc`: root, length, shift) | `recurrence/lte_gold` | `rp.sequence.load("lte_gold", c_init=77)` | [sequence/\_\_init\_\_.py](https://github.com/Anonoei/RPPS/blob/main/src/rpps/sequence/__init__.py) |
| `scramble` | how a sequence is applied: `additive` (names a sequence), `feedthrough` (self-synchronizing) | `additive/dvb_energy` | `rp.code.scram.load("dvb_energy")` | [code/scram/serial.py](https://github.com/Anonoei/RPPS/blob/main/src/rpps/code/scram/serial.py) |
| `fec` | block and convolutional codes (LDPC from DVB-S2 tables, dense or sparse checks, or a quasi-cyclic matrix of shifts: `ldpc/ieee80211n_648_1_2`) | `block/rs/204_188` | `rp.code.load("block/rs/204_188")` | [code/serial.py](https://github.com/Anonoei/RPPS/blob/main/src/rpps/code/serial.py) |
| `interleave` | block and Forney interleavers | `forney/12_17` | `rp.code.load("interleave/forney/12_17")` | [code/serial.py](https://github.com/Anonoei/RPPS/blob/main/src/rpps/code/serial.py) |
| `crc` | CRCs (reveng's fields) | `crc32_mpeg2` | `rp.link.load("crc/crc32_mpeg2")` | [link/serial.py](https://github.com/Anonoei/RPPS/blob/main/src/rpps/link/serial.py) |
| `frame` | frame formats (HDLC) | `hdlc` | `rp.link.load("frame/hdlc")` | [link/serial.py](https://github.com/Anonoei/RPPS/blob/main/src/rpps/link/serial.py) |
| `header` | `physical` (known symbols, a short code carrying fields) and `link` (fields, CRC, payload length) | `physical/dvbs2_pl` | `rp.symbol.load_header("dvbs2_pl")`, `rp.link.load("header/ieee80211b_plcp")` | [symbol/frame.py](https://github.com/Anonoei/RPPS/blob/main/src/rpps/symbol/frame.py), [link/header.py](https://github.com/Anonoei/RPPS/blob/main/src/rpps/link/header.py) |
| `table` | rows of parameters looked up by header fields | `dvbs2/modcod` | `rp.profile.table("dvbs2/modcod")` | [profile/control.py](https://github.com/Anonoei/RPPS/blob/main/src/rpps/profile/control.py) |
| `plan` | burst time plans: superframes, frames, timeslots, burst types, assignments, hopping (DVB-RCS2's model) | `demo/mftdma` | `rp.plan.load("demo/mftdma")` | [plan/\_\_init\_\_.py](https://github.com/Anonoei/RPPS/blob/main/src/rpps/plan/__init__.py) |
| `profile` | whole signals | `dvbs2/normal` | `rp.profile.load("dvbs2/normal")` | [profile/steps/](https://github.com/Anonoei/RPPS/tree/main/src/rpps/profile/steps), `controls`: [profile/control.py](https://github.com/Anonoei/RPPS/blob/main/src/rpps/profile/control.py) |
| `system` | named radio systems by what blind tools measure (`"features"`: modulation, baud, roll-off, frame, bands...; the `"profile"` that decodes one), what `rp.analysis.catalogue.match` scores against | `satellite/dvbs2` | `rp.analysis.catalogue.systems()` | [analysis/catalogue.py](https://github.com/Anonoei/RPPS/blob/main/src/rpps/analysis/catalogue.py) |

`python -m rpps ls` lists what is available (`python -m rpps ls sequence scramble` for some).

A definition names the others it uses by name: an additive scrambler its sequence (`"sequence": {"name": "lte_gold", "args":
{"c_init": {"expr": "..."}}}`), a frame sync a header, a profile everything. So one sequence serves a scrambler, a spreader,
a preamble search and a phase scrambler alike.

## Names
A definition is referenced by its path below the category, without `.json`, case-insensitive: `rs/204_188`, `k7_171_133/3_4`.
- The folders say what it is, so the filename holds only the parameters. A group folder is the structure of the thing (`rs`, `ldpc`, `k7_171_133`); the standard goes in `ref` / `comment`, and in `profile/<family>/`.
- A bare name (`204_188`) works while exactly one file has it; otherwise the error lists the choices.
- Definitions that use others (profiles above all) write the path, so the reader sees what each is: `"additive/dvb_energy"`
  (a sequence XOR'd in: it must start on a frame boundary), `"feedthrough/ieee80211b"` (self-synchronizing: each channel
  error comes out once per tap), `"rs/204_188"`. Bare names are for typing at a prompt: `rp.code.scram.load("v35")`.
- Other names are not kept: what used to be an alias (`dvbs.3_4_7`, `adsb`) is a tag now, see below.

## Parents and children
A definition can be another one with a few keys changed: `"extends"` names it, in the same resource (looked up below the
child's own category first, so `"extends": "rs/p25"` in fec/block works). The child holds only what differs:
```json
{"base": "constellation", "extends": "apsk/dvbs2_16apsk", "comment": "DVB-S2 16APSK for code rate 3/4: gamma = 2.85",
 "params": {"gamma": 2.85}}
```
- Dicts merge key by key, lists and values replace, `null` deletes a key the parent has.
- A child profile changes single steps with `"steps"` as a dict, by step id (its `"id"`, else its kind: `"timing"`):
  `{"shape": {"pulse": "rrc/0p20"}, "carrier": null, "+after:timing": [{"sync": "agc"}], "+end": [...]}` merges into a
  step, takes one out and puts steps in (`+before:<id>`, `+after:<id>`, `+start`, `+end`). Two steps with the same id
  can't be patched until they have `"id"`s; a list still replaces them all.
- `comment` and `ref` are each file's own; `tags` add to the parent's; `params` merge by name, and a bare value pins the
  parent's default (`"params": {"code": "k7_171_133/3_4"}`).
- `"abstract": true` marks a parent that is not a definition on its own (a family's shared fields: the mother codes
  `conv/k7_171_133`..., `ldpc/dvbs2_64800`, `bch/dvbs2_normal`, `rs/p25`). `ls` hides it (`python -m rpps ls --all` shows it) and loading it is an error.
- Chains are fine; a loop, or a parent in another resource, is an error.
- `python -m rpps show <resource>/<ref>` prints a definition with its parents applied.

**Param or child?** A choice made when loading (samples per symbol, a cell id, which punctured rate a DVB-S receiver
runs) is a param; one that changes several steps together is a param with choices (below). A variant a standard or a table names (16APSK for rate 3/4, the code of MODCOD 7) is a child, so the
name keeps working. A child that overrides most of its parent is probably not one.

Families written this way: `constellation/psk` (QPSK, 8PSK as rings; the DVB-S2 labellings and `qpsk_axes` are children),
`constellation/apsk` (16/32APSK with the radii as params, one child per code rate), `fec/conv/<mother>.json` (the
generators; every rate in `<mother>/`, the unpunctured 1/n too, is a child adding its `puncture`), `fec/block/ldpc`, `bch`, `rs/p25*`,
`interleave/block/dvbs2_8psk_210`, `profile/dvbs/qpsk` (the rate a `code` param, a child per rate).

## Variants: choices and "when"
A param can list its `"choices"`, each a bundle of fields that one choice sets together; a step names a field as
`{"param": "modulation.shape"}` (or `modulation.shape` in an expression), and a list item with `"when"` is kept only
where its expression holds:
```json
"params": {"modulation": {"default": "qpsk", "choices": {"qpsk": {"shape": "QPSK", "threshold": 0.6},
                                                         "16qam": {"shape": "16QAM", "threshold": 0.85}}},
           "fec": {"default": "hamming", "choices": ["hamming", "none"]}},
"steps": [..., {"symbol": "constellation", "shape": {"param": "modulation.shape"}}, ...,
          {"code": "block", "name": "hamming/7_4", "when": "fec == 'hamming'"}]
```
- A value that is not one of the choices is an error naming them; `"choices": [...]` (a list) only checks the value.
- A choice matches its key or the key's text (`0.2` is `"0.2"`: JSON keys are strings).
- Expressions take string literals, so `"when": "modulation == '16qam'"` and `"expr": "... if fec == 'none' else ..."` work.
- A child's `choices` replace its parent's, so a child can narrow them. A mode a standard names stays a child that pins
  the default (`dvbs/qpsk_1_2`), so the name keeps working.
- Blind analysis (`rp.analysis.identify`, `rpps solve`, the survey's framed profiles) tries every variant the choices
  make (`rp.profile.variants(name)`, the defaults first) and names the one that fits with its params
  (`demo/variants (modulation=16qam, rolloff=0.2)`); `"blind": false` on a param (a payload size) keeps it at its default.
- Expressions join strings with `+` (`{"expr": "'k7_171_133/' + rate"}`) and take `min`, `max`, `abs`, `len`, `ceil`,
  `floor`, `round` and `log2`.
- A param may say its `"type"` (`int`, `float`, `str`, `bool`), `"min"` and `"max"`: `rpps rx -p sps=fast` is then an
  error when the profile loads, not a failure in a step.
- `rp.profile.info(name, **params)` evaluates `requires` with the chosen bundle (`"bw": "(1 + rolloff)*baud"`) and lists
  each param's choices.

A check that is not one CRC is a `"composite"` crc definition: CRC parts over transformed views of the message
(`nibble_swap`, `reverse_bits`, `reverse_bytes`, `invert`, `xor`, `skip`, `take`, and your own through
`rp.link.check.register`), concatenated or XORed. It goes wherever a CRC does: an HDLC frame's `fcs`, a link header's `crc` or
`fcs` ([`rpps.link`](../api/link.md)).

```python
--8<-- "examples/profile_variants.py"
```

## Fields
What a frame's bits say is a list of fields ([`rpps.link.fields`](../api/link.md)), in link headers (`"fields"`), in
headers' and frames' `"payload_fields"` (a message's `fields["payload"]`) and in a profile step's `"message"`:
```json
"payload_fields": [
  {"name": "addresses", "until": "ext == 1", "fields": [
    {"name": "call", "type": "text", "bytes": 6, "shift": 1},
    {"name": "ssid", "from": 51, "bits": 4}, {"name": "ext", "bits": 1}]},
  {"name": "control", "bits": 8},
  {"name": "pid", "bits": 8, "if": "(control & 1) == 0", "enum": {"240": "no layer 3"}}]
```
- A field is an unsigned integer (`bits`, an expression over the fields before it too), `signed`, scaled (`scale`,
  `offset`), with its own bit `order` and `enum` names (`pid.name`); or `"type": "bytes"` / `"text"` (`bytes`,
  `reverse`, `shift`).
- `"if"` reads it only where an expression over the fields before holds, decided frame by frame (a definition's `"when"`
  is decided at load, over its params); `"from"` reads it at a bit position.
- A field with `"fields"` is a group: a dict, or a list with `"repeat"` (a count, an expression, or `"*"`: to the end) or
  `"until"` (an expression over each element). Expressions see `left`, the bits not yet read.
- A link header whose fields are conditional or sized by expressions is read as it comes, however the bits are cut.
- `frame/ax25` (addresses, control, PID), `header/ble_adv` (PDU type names, AdvA, AD structures) and `dvbs/qpsk` (the TS
  adaptation field and PCR) are written this way.

## Codes bit by bit
A block code whose standard writes its parity bits as equations, puts its bits in its own order or inverts some of them
is still a definition. Any block code (`fec/block`) takes, after its own keys:

- `"append": [[14, 15]]`: check bits after the codeword, each the XOR of the codeword bits listed (P25's NID: BCH (63,16)
  and a 64th bit that is S1 xor S0 of the DUID, `bch/p25_nid`). `"extend": true` is the overall parity bit.
- `"layout"`: where each codeword bit is sent, a list of positions or `{"message": [...], "parity": "rest"}`.
- `"xor"`: a pattern on the word sent (a number of n bits, or a list), for inverted or masked parity.

Decoding undoes them in reverse; an appended bit the decision disagrees with counts as one more error, and a word past
what the code corrects is a failure. A `linear` code can give its parity bits as the message bits each one XORs, the way
standards write them:

```json
{"base": "block", "type": "linear", "n": 7, "k": 4, "parity": [[0, 1, 3], [0, 2, 3], [1, 2, 3]],
 "layout": {"message": [2, 4, 5, 6], "parity": "rest"}}
```

A short code given only as what each bit is, `"type": "expr"`, writes one expression per codeword bit over the message
bits `m0..` (and the bits before it, `c0..`), with `^ & | ~`. One that is linear (or linear and inverted) loads as a
linear code in its own layout and decodes on its syndrome trellis, soft; any other decodes against its whole codebook
(k up to 16):

```json
{"base": "block", "type": "expr", "k": 4, "bits": ["m0 ^ m1 ^ m3", "m0 ^ m2 ^ m3 ^ 1", "m0", "m1 ^ m2 ^ m3", "m1", "m2", "m3"]}
```

A trellis code (`fsm`) with a `length` (message bits a codeword) is one codeword per block, each started and flushed on
its own: `trellis/p25_1_2` takes 96 bits to 196.

## Tags
`"tags": ["dvb-s2"]` says which standards or systems use a definition. The folders say what a thing is; tags say who uses
it; `ref` cites the clause. Tags come from [tags.json](https://github.com/Anonoei/RPPS/blob/main/config/tags.json) (a tag not listed there fails the tests), and children
inherit their parent's.
```
python -m rpps ls --tags              # every tag and how many definitions carry it
python -m rpps ls --tag dvb-s2        # constellations, codes, interleavers, header, table and profile of DVB-S2
rp.tagged("ccsds")                    # {resource: [refs]}
```

## Differential modulation
There is no DPSK definition: a differential step chains with a plain constellation, whose points are then the phase
changes. pi/4-DQPSK is `psk/qpsk` after `{"symbol": "differential"}`; 802.11b DQPSK is `psk/qpsk_axes`; DBPSK is
`psk/bpsk`. Bit-level differential coding (NRZ-M, NRZ-I) is `{"code": "line"}`.

## File header
Every file starts with `base` (the category, picks the loader), then `extends` and `abstract` where it has them, `type` (the implementation, where there are several), `comment`, and optionally `ref` (the spec) and `tags`. `python -m rpps fmt` puts files in the canonical style (`--check` only lists differences).

## Keys nothing reads
A key that nothing reads is reported when the definition or profile loads, as an `rp.errors.ConfigWarning` naming it:
a misspelling (`"cutof"`), or a setting the method it names does not take (a loop's `"alpha"` on a block estimator).
For a profile that covers its own keys, every step's, its `defaults` (a kind with no step to take them is reported
too) and the `sync` settings given to `rp.profile.load`. When `sync` replaces a step's method, that step's own settings
were for the method it named, and are not reported. A loader that designs something later (a filter once the stream's
rate is known) takes the keys it will use when it loads, so they count. The tests turn the warning into an error.

## Adding
- **A definition**: drop a JSON in the right folder. A variant of one that exists: a child that `extends` it. A tag it needs: add it to [tags.json](https://github.com/Anonoei/RPPS/blob/main/config/tags.json).
- **Your own**: keep a folder laid out like this one and call `rp.add_config_path(dir)` or set `$RPPS_CONFIG`. It is searched first, so it can also override shipped definitions.
- **A category**: add the folder and `_config.register_domain("<domain>").register("<category>", loader)`, where `loader(name, obj)` builds the object (reading, through `obj`, every key it uses: what it never reads is reported).
- **A resource**: `_config.register_domain("<name>", {"<base>": loader})`; `_config.ls / find / load` then work on it. [link/serial.py](https://github.com/Anonoei/RPPS/blob/main/src/rpps/link/serial.py) is the smallest example.

## Bringing in existing data
Definitions never point at other files. Convert instead; the result is a normal definition:
```python
# docs: skip
rp.signal.filters.import_fir("taps.f32", dtype="f32", name="my_lp", dest="mine")     # also .txt/.csv, .npy/.npz, C/Python arrays, Xilinx .coe
rp.signal.filters.import_iir("sections.npy", form="sos", name="my_iir", dest="mine") # form: sos | ba | zpk
rp.code.import_alist("code.alist", "mine", dest="mine")                     # LDPC, MacKay alist
rp.add_config_path("mine")
```
Without `name` they return the dict. Arrays in memory go through `fir_def`, `sos_def`, `ba_def`, `zpk_def`. An existing file is only replaced with `overwrite=True`.

## Coming from GNU Radio
| GNU Radio | Here |
|---|---|
| `fec` LDPC `.alist` | `import_alist` |
| `digital.crc`, reveng entries | `crc`: reveng's fields; `check` is verified on load |
| `additive_scrambler`, CCSDS, DVB | `sequence/recurrence` (the register as a recurrence: `taps`, `init`) and `scramble/additive` naming it (`frame` = restart structure, `period`) |
| `scrambler_bb`, G3RUH, V.35 | `scramble/feedthrough` (`register`: fibonacci, galois_left or galois_right; `poly` = the polynomial's exponents, 1 + x^14 + x^17 is [14, 17], as the standards write it; `seed`). A recurrence's `taps` are positions in its window instead, from 0 |
| `cc_encoder_def(polys, k, rate)` | `rp.code.Conv(...)`, or the mother code `conv/k<K>_<g1>_<g2>` and its punctured rates `conv/k<K>_<g1>_<g2>/<rate>`; generators are bit lists (`0o171` is `[1,1,1,1,0,0,1]`), the first bit multiplying the current input |
| `firdes.low_pass(gain, Fs, cutoff, transition, window)` | `fir` `windowed`: `cutoff`, `transition`, `atten_db`, `units: "hz"`, `gain` |
| `firdes.root_raised_cosine` | `pulse/rrc/<alpha>` |
| `fir_filter_ccf` | `fir` `taps` (complex: `taps_imag`) |
| `iir_filter_ffd(fftaps, fbtaps)` | `ba_def(fftaps, fbtaps, convention="gnuradio")`. Default `oldstyle=True` adds the feedback terms, so `a = [1, -fbtaps[1:]]`; `oldstyle=False` is scipy's |
| `single_pole_iir_filter_ff` | `iir` `single_pole` |
| `differential_encoder_bb`, NRZI, Manchester | profile step `{"code": "line", "name": "nrzm" \| "nrzi" \| "manchester" \| "thomas" \| "invert"}` (`rp.code.Line`) |
| `correlate_access_code`, sync words | `sequence/word` plus `{"sync": "delimit", "sequence": ...}` to find it and `{"code": "marker", "sequence": ...}` to attach / strip it |
| `conjugate_cc` (spectral inversion) | `{"signal": "conjugate"}`; or a constellation map "inverted" at the symbol level |
| `freq_xlating_fir_filter` + decimation | `{"signal": "channelize", "freq": ..., "width": ...}` (designed from the stream's rate) |
| `rational_resampler` | `{"signal": "resample", "up": ..., "down": ...}` |
| `quadrature_demod` / `frequency_modulator` | `{"wave": "fm", "h": ..., "sps": ...}` (FSK, GFSK, GMSK) or `{"wave": "fm", "deviation": ...}` (analog) |
| `am_demod`, `ssb` | `{"wave": "am"}`, `{"wave": "ssb", "bw": ..., "sideband": ...}` |
| `hdlc_framer` / `hdlc_deframer` | `frame/hdlc` and `{"link": "frame", "name": "hdlc"}` (streaming, any block cut) |
| `packet_headergenerator` + CRC trailer | `header/link` with `fields`, `payload` and `fcs` (a CRC over header and payload after the payload) |

Anything GNU Radio builds from parameters can be built without a file: `rp.code.Hamming/RS/Repetition/Conv`, or `rp.code.init(name, dict)`.
Not expressible yet: a frequency-hopping schedule.
