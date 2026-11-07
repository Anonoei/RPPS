# config/

Definitions RPPS loads by name. Layout: `config/<domain>/<category>/[<group>/]<name>.json`

| Domain | Categories (`base`) | Example | Loaded by |
|---|---|---|---|
| `mod` | `psk` | `mod/psk/qpsk.json` | `rp.mod.load("QPSK")` (also generates `16QAM`, `8PSK`... with no file) |
| `scram` | `fdt`, `adt` | `scram/adt/dvb_energy.json` | `rp.scram.load("dvb_energy")` |
| `coding` | `block`, `conv`, `interleave` | `coding/block/rs/204_188.json` | `rp.coding.load("block", "rs/204_188")` |
| `encap` | `crc`, `frame` | `encap/crc/crc32_mpeg2.json` | `rp.encap.load("crc", "crc32_mpeg2")` |
| `chain` | `chain` | `chain/dvbs/3_4.json` | `rp.Chain.load("dvbs/3_4")` |

## Names
A definition is referred to by its path below the category, without `.json`, case-insensitive: `rs/204_188`, `hamming/7_4`, `k7_171_133/3_4`. The folder says what it is, so the filename only carries the parameters.
- The folder under a category is the **structure** of the code (`rs`, `ldpc`, `k7_171_133` = constraint length + generators). The standard goes in `ref` and `comment`, and in `chain/<standard>/`, which ties a standard's pieces together.
- A bare name (`204_188`, `dvb_energy`) works when exactly one file has it, and raises an "ambiguous" error listing the choices otherwise.
- `"aliases": [...]` in a file gives it other names. The old `family.name` form (`rs.204_188`) still resolves too. Both log a deprecation warning once.
- The folder before the name is also what the loaded object is called (`rs`, `k7_171_133`).

## File header
Every file starts with the same keys:
- `base`: the category. Picks the loader.
- `type`: the implementation inside the category (where there is more than one).
- `comment`: one line on what it is. `ref`: the spec, optionally. `aliases`: other names it answers to, optionally.

Keep files tidy with `python tests/dev/fmt_config.py` (`--check` to only list differences).

## Adding things
- **A definition**: drop a JSON in the right folder. It appears in `ls()` and loads by name.
- **Your own definitions**: keep a folder laid out like this one and pass it to `rp.add_config_path(dir)` or set `$RPPS_CONFIG`. It is searched first, so it can also override shipped ones.
- **A category**: make `config/<domain>/<category>/` and register its loader, `_config.register_domain("<domain>").register("<category>", loader)` where `loader(name, obj)` builds the object. `ls`, `find` and `load` need nothing else.
- **A domain**: `_config.register_domain("<name>", {"<category>": loader})`, then use `_config.ls / find / load("<name>", ...)`. See `rpps/encap/serial.py` for the smallest example.

## Coming from GNU Radio
Files are named presets of parameters. Anything GNU Radio builds from parameters can also be built here without a file: `rp.coding.Hamming/RS/Repetition/Conv`, or `rp.coding.init(name, dict)` with the same dict a file holds.

| GNU Radio | Here |
|---|---|
| `fec` LDPC `.alist` H matrix | `rp.coding.import_alist("code.alist", "mine")` writes a normal `block/ldpc` definition |
| `digital.crc` / reveng catalogue entry | `encap/crc/*.json`: `width, poly, init, refin, refout, xorout, check` are reveng's fields, and `check` is verified on load |
| `additive_scrambler(mask, seed, len)`, CCSDS, DVB | `scram/adt`: `poly` are the feedback taps, `seed` the register, `frame` the restart period |
| `scrambler_bb` / G3RUH / V.35 | `scram/fdt` (self-synchronizing) |
| `cc_encoder_def(polys, k, rate)` | `rp.coding.Conv(...)` or a `conv/k<K>_<g1>_<g2>/<rate>.json` file; generators are bit lists |

Not expressible yet (add as needed): convolutional tail-biting/truncated modes and per-generator inversion (CCSDS G2), Reed-Solomon dual-basis (CCSDS), QC-LDPC lifting (5G, Wi-Fi), differential constellations, FSK/GMSK/OQPSK definitions, sync-word and line-code domains. Polynomials are bit lists, not octal ints: `0o171` is `[1,1,1,1,0,0,1]`.
