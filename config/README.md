# config/

JSON definitions RPPS loads by name, one folder per **resource**: what a definition is, not which package loads it
(ARCHITECTURE.md, Config resources). `config/<resource>/[<group>/]<name>.json`. A file's `base` picks the loader within
the resource and its `type` the implementation; the keys each accepts are in the header of the module linked.

| Resource | Holds | Example | Load | Keys |
|---|---|---|---|---|
| `constellation` | point sets and their bit maps | `psk/qpsk` | `rp.symbol.load("QPSK")` | [symbol/constellation.py](../src/rpps/symbol/constellation.py) |
| `pulse` | symbol and frequency pulses (`"kind"`) | `rrc/0p35` | `rp.signal.filters.load("pulse", "rrc/0p35")` | [signal/filters/serial.py](../src/rpps/signal/filters/serial.py) |
| `filter` | FIR, IIR, windows | `fir/windowed/lowpass_0p1_60db` | `rp.signal.filters.load("fir", "windowed/lowpass_0p1_60db")` | [signal/filters/serial.py](../src/rpps/signal/filters/serial.py) |
| `sequence` | bit sequences (recurrences, Gold) and chip codes (Barker, PN sets, GPS C/A) | `recurrence/lte_gold` | `rp.sequence.load("lte_gold", c_init=77)` | [sequence/\_\_init\_\_.py](../src/rpps/sequence/__init__.py) |
| `scramble` | how a sequence is applied: `additive` (names a sequence), `feedthrough` (self-synchronizing) | `additive/dvb_energy` | `rp.code.scram.load("dvb_energy")` | [code/scram/serial.py](../src/rpps/code/scram/serial.py) |
| `fec` | block and convolutional codes | `block/rs/204_188` | `rp.code.load("block", "rs/204_188")` | [code/serial.py](../src/rpps/code/serial.py) |
| `interleave` | block and Forney interleavers | `forney/12_17` | `rp.code.load("interleave", "forney/12_17")` | [code/serial.py](../src/rpps/code/serial.py) |
| `crc` | CRCs (reveng's fields) | `crc32_mpeg2` | `rp.link.load("crc", "crc32_mpeg2")` | [link/serial.py](../src/rpps/link/serial.py) |
| `frame` | frame formats (HDLC) | `hdlc` | `rp.link.load("frame", "hdlc")` | [link/serial.py](../src/rpps/link/serial.py) |
| `header` | `physical` (known symbols, a short code carrying fields) and `link` (fields, CRC, payload length) | `physical/dvbs2_pl` | `rp.symbol.load_header("dvbs2_pl")`, `rp.link.load("header", "ieee80211b_plcp")` | [symbol/frame.py](../src/rpps/symbol/frame.py), [link/header.py](../src/rpps/link/header.py) |
| `table` | rows of parameters looked up by header fields | `dvbs2/modcod` | `rp.profile.table("dvbs2/modcod")` | [profile/control.py](../src/rpps/profile/control.py) |
| `profile` | whole signals | `dvbs2/normal` | `rp.profile.load("dvbs2/normal")` | [profile/steps.py](../src/rpps/profile/steps.py), `controls`: [profile/control.py](../src/rpps/profile/control.py) |

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
- `"aliases": [...]` gives a file other names. The old `family.name` form (`rs.204_188`) still resolves. Both warn once.

## File header
Every file starts with `base` (the category, picks the loader), `type` (the implementation, where there are several), `comment`, and optionally `ref` (the spec) and `aliases`. `python -m rpps fmt` puts files in the canonical style (`--check` only lists differences).

## Adding
- **A definition**: drop a JSON in the right folder.
- **Your own**: keep a folder laid out like this one and call `rp.add_config_path(dir)` or set `$RPPS_CONFIG`. It is searched first, so it can also override shipped definitions.
- **A category**: add the folder and `_config.register_domain("<domain>").register("<category>", loader)`, where `loader(name, obj)` builds the object.
- **A resource**: `_config.register_domain("<name>", {"<base>": loader})`; `_config.ls / find / load` then work on it. [link/serial.py](../src/rpps/link/serial.py) is the smallest example.

## Bringing in existing data
Definitions never point at other files. Convert instead; the result is a normal definition:
```python
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
| `scrambler_bb`, G3RUH, V.35 | `scramble/feedthrough` (`register`, `poly` = feedback taps, `seed`) |
| `cc_encoder_def(polys, k, rate)` | `rp.code.Conv(...)` or `conv/k<K>_<g1>_<g2>/<rate>`; generators are bit lists (`0o171` is `[1,1,1,1,0,0,1]`) |
| `firdes.low_pass(gain, Fs, cutoff, transition, window)` | `fir` `windowed`: `cutoff`, `transition`, `atten_db`, `units: "hz"`, `gain` |
| `firdes.root_raised_cosine` | `shaping/rrc/<alpha>` |
| `fir_filter_ccf` | `fir` `taps` (complex: `taps_imag`) |
| `iir_filter_ffd(fftaps, fbtaps)` | `ba_def(fftaps, fbtaps, convention="gnuradio")`. Default `oldstyle=True` adds the feedback terms, so `a = [1, -fbtaps[1:]]`; `oldstyle=False` is scipy's |
| `single_pole_iir_filter_ff` | `iir` `single_pole` |

Anything GNU Radio builds from parameters can be built without a file: `rp.code.Hamming/RS/Repetition/Conv`, or `rp.code.init(name, dict)`.
Not expressible yet: convolutional tail-biting and per-generator inversion (CCSDS G2), Reed-Solomon dual basis (CCSDS), QC-LDPC lifting, differential constellations, FSK/GMSK/OQPSK definitions, sync-word and line-code domains.
