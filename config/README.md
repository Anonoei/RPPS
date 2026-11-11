# config/

JSON definitions RPPS loads by name: `config/<domain>/<category>/[<group>/]<name>.json`. The keys each domain accepts are documented
in the header of its `serial.py`.

| Domain | Categories | Example | Load | Keys |
|---|---|---|---|---|
| `mod` | `constellation`, `header` | `constellation/psk/qpsk`, `header/dvbs2_pl` | `rp.symbol.load("QPSK")` | [mod/serial.py](../src/rpps/mod/serial.py) |
| `spread` | `fixed`, `pnset`, `gold` | `fixed/barker11` | `rp.spread.load("gps_ca", prn=1)` | [spread/code.py](../src/rpps/spread/code.py) |
| `scram` | `fdt`, `adt` | `adt/dvb_energy` | `rp.code.scram.load("dvb_energy")` | [scram/serial.py](../src/rpps/scram/serial.py) |
| `coding` | `block`, `conv`, `interleave` | `block/rs/204_188` | `rp.code.load("block", "rs/204_188")` | [coding/serial.py](../src/rpps/coding/serial.py) |
| `encap` | `crc`, `frame`, `header` | `crc/crc32_mpeg2`, `header/ieee80211b_plcp` | `rp.link.load("crc", "crc32_mpeg2")` | [encap/serial.py](../src/rpps/encap/serial.py) |
| `filters` | `fir`, `iir`, `shaping`, `window` | `shaping/rrc/0p35` | `rp.signal.filters.load("shaping", "rrc/0p35")` | [filters/serial.py](../src/rpps/filters/serial.py) |
| `profile` | `profile` | `dvbs/qpsk_3_4` | `rp.profile.load("dvbs/qpsk_3_4")` | [profile/steps.py](../src/rpps/profile/steps.py), `controls`: [profile/control.py](../src/rpps/profile/control.py) |
| `table` | `table` | `dvbs2/modcod` | `rp.profile.table("dvbs2/modcod")` | [profile/control.py](../src/rpps/profile/control.py) |

`rp.<domain>.ls()` lists what is available.

## Names
A definition is referenced by its path below the category, without `.json`, case-insensitive: `rs/204_188`, `k7_171_133/3_4`.
- The folders say what it is, so the filename holds only the parameters. A group folder is the structure of the thing (`rs`, `ldpc`, `k7_171_133`); the standard goes in `ref` / `comment`, and in `profile/<family>/`.
- A bare name (`204_188`) works while exactly one file has it; otherwise the error lists the choices.
- `"aliases": [...]` gives a file other names. The old `family.name` form (`rs.204_188`) still resolves. Both warn once.

## File header
Every file starts with `base` (the category, picks the loader), `type` (the implementation, where there are several), `comment`, and optionally `ref` (the spec) and `aliases`. `python -m rpps fmt` puts files in the canonical style (`--check` only lists differences).

## Adding
- **A definition**: drop a JSON in the right folder.
- **Your own**: keep a folder laid out like this one and call `rp.add_config_path(dir)` or set `$RPPS_CONFIG`. It is searched first, so it can also override shipped definitions.
- **A category**: add the folder and `_config.register_domain("<domain>").register("<category>", loader)`, where `loader(name, obj)` builds the object.
- **A domain**: `_config.register_domain("<name>", {"<category>": loader})`; `_config.ls / find / load` then work on it. [encap/serial.py](../src/rpps/encap/serial.py) is the smallest example.

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
| `digital.crc`, reveng entries | `encap/crc`: reveng's fields; `check` is verified on load |
| `additive_scrambler`, CCSDS, DVB | `scram/adt` (`poly` = feedback taps, `seed`, `frame` = restart period) |
| `scrambler_bb`, G3RUH, V.35 | `scram/fdt` |
| `cc_encoder_def(polys, k, rate)` | `rp.code.Conv(...)` or `conv/k<K>_<g1>_<g2>/<rate>`; generators are bit lists (`0o171` is `[1,1,1,1,0,0,1]`) |
| `firdes.low_pass(gain, Fs, cutoff, transition, window)` | `fir` `windowed`: `cutoff`, `transition`, `atten_db`, `units: "hz"`, `gain` |
| `firdes.root_raised_cosine` | `shaping/rrc/<alpha>` |
| `fir_filter_ccf` | `fir` `taps` (complex: `taps_imag`) |
| `iir_filter_ffd(fftaps, fbtaps)` | `ba_def(fftaps, fbtaps, convention="gnuradio")`. Default `oldstyle=True` adds the feedback terms, so `a = [1, -fbtaps[1:]]`; `oldstyle=False` is scipy's |
| `single_pole_iir_filter_ff` | `iir` `single_pole` |

Anything GNU Radio builds from parameters can be built without a file: `rp.code.Hamming/RS/Repetition/Conv`, or `rp.code.init(name, dict)`.
Not expressible yet: convolutional tail-biting and per-generator inversion (CCSDS G2), Reed-Solomon dual basis (CCSDS), QC-LDPC lifting, differential constellations, FSK/GMSK/OQPSK definitions, sync-word and line-code domains.
