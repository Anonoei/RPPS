# AGENTS.md

Read this first: it says where everything is.

RPPS (RF Parameter Processor Suite) is a Python library (numpy, optional C++ kernels) that generates and processes radio
signals: filters, sync, modulation, FEC, framing, and whole standards (DVB-S/S2, 802.11b, 802.15.4, BLE, ADS-B, CCSDS,
LTE and 5G NR to SIB1) as JSON profiles, plus blind analysis of unknown signals. It stays **signal agnostic**: a new
signal should be new definitions plus existing operators.

## Commands

```
python tools/builder.py -s      # pip install -e ".[dev]"
python tools/builder.py -c      # build the C++ kernels in place (rpps._native)
python -m pytest -n auto -m "not slow"                                     # what CI runs
RPPS_NO_NATIVE=1 python -m pytest -n auto -m "not slow and not kernels"    # the Python loops
python tools/builder.py -d      # docs: API pages + llms files (tools/gen_docs.py), then zensical build --strict
python -m rpps ls [resource]    # shipped definitions; also: show, steps, rx, tx, survey, profile, solve, fmt
```

A stale `rpps._native` warns at import: rebuild (`-c`) after touching `csrc/`.

## The model in one screen

```
samples -> symbols -> channel bits -> data bits -> packets
 signal (conditioning: samples -> samples)        sync (estimate + correct, at any representation; never in the data)
 wave (samples <-> symbols) | symbol (symbols <-> channel bits, LLRs on receive) | code (channel <-> data bits) | link (data bits <-> packets)
```

- Operators: `data * pipe` encodes, `data / pipe` decodes (`Pipe`: symbol, code, link); `data + stage` processes (`Stage`:
  signal, sync, wave); `pipe * pipe` / `stage + stage` make a `Pipeline`; `data >> sink` writes.
- `rp.Data` wraps an array with `Meta` (`Fs`, `sps`, `cf`, absolute position `meta.pos`, the stream). A `Stage` keeps
  per-stream state in `new_state(meta)`, held by `stream`; a `Pipe` keeps leftovers in `ctx`. Output must not depend on
  how the input is cut into blocks (any length, including 0).
- Form vs representation: `dtype` (SAMPLES, REAL, SYMBOLS, LLR, BITS, BYTES, MSG) and `Rep` (SAMPLES, SYMBOLS, CHANNEL,
  DATA, PACKETS). Operators check `takes()`; a link refuses channel bits (`rp.code.Uncoded()` between them).
- LLRs are log(P0/P1) (positive = 0). Bits are MSB first. `bool` arrays are bits, `uint8`/`bytes` are bytes.
- **Where a function goes is decided by what it takes and gives**, not what a standard calls it: estimating an unknown is
  sync; inserting/removing known structure belongs to the transform where it lives; finding where it starts is sync.
- Sync: block estimation over windows at absolute positions, tracked across windows (`sync.track`), is first class;
  per-sample loops (Costas, M&M, LMS) are second class.
- A profile (`config/profile/<family>/<mode>.json`) lists a signal's steps in receive order; `p.rx()` runs them,
  `p.tx()` runs the transforms inverted in reverse. Headers drive later steps through `controls` (positions, not time).

Full design: the `rpps.base` docstring (model, rule, categories, contracts, grey areas, type tags, streaming rules).

## Where things are

| Path | What |
|---|---|
| `src/rpps/<pkg>/` | one package per category: source, signal, sync, wave, symbol, code, link, grid, profile, parallel, sequence, plan, analysis, base, math, utils, viz |
| `src/rpps/<pkg>/__init__.py` | **the package's documentation** (its docstring), rendered in the API reference |
| `config/<resource>/` | JSON definitions: constellation, pulse, filter, sequence, scramble, fec, interleave, crc, frame, header, linktype, table, plan, profile; `tags.json` |
| `csrc/rpps_native/` | C++17 kernels (pybind11); the Python loop in the calling stage is the reference |
| `docs/` | the site: guides (`docs/guide/`), examples index, contributing, roadmap; `docs/api/` and `docs/llms*.txt` are generated |
| `tools/` | `builder.py` (setup, kernels, tests, docs, release), `gen_docs.py` (API pages, llms files), benchmarks (`*_bench.py`, `tpc_ber.py`) |
| `examples/` | one self-checking script per recommended usage; the guides embed them and the tests run them |
| `tests/unit/` | a folder per package, plus `standards/`, `policy/`, `native/`; helpers in `tests/support/` |
| `projects/` | worked receivers on real signals, each with a README |
| `private/` | untracked captures and tooling: **never commit or quote it** |

Imports run down the chain: nothing in the data path imports `analysis`, `profile` or `parallel` (`test_layering.py`).
Nothing under `src/` may import scipy (`test_numpy_only.py`).

## Where to read more

| Question | Read |
|---|---|
| What package X holds, and its rules | `src/rpps/X/__init__.py` docstring (`help(rpps.X)`) |
| Sync methods, feedforward rules | `rpps.sync` |
| Profile features (bursts, branches, grids, messages, kinds, requires) | `rpps.profile`; header-driven control: `rpps.profile.control` |
| Step kinds a profile can name | `rpps.profile.steps`, `python -m rpps steps` |
| Definition format, names, `extends`, tags | `docs/guide/definitions.md` |
| Which blind tool answers which question | `rpps.analysis` |
| Parallel region, batch kernels | `rpps.parallel` |
| Effort levels and measured gains | `rpps._effort` |
| How to use it for a task | `docs/guide/*.md` |
| Setup, test markers, kernel pitfalls, CI, releases | `docs/contributing.md` |
| Everything at once | `docs/llms-full.txt` (after `python tools/gen_docs.py`) |

## Conventions

- **Operators first:** keep features inside `*`, `/`, `+`, `>>` rather than adding setup calls.
- **No aliases:** a rename or move drops the old name and fixes the callers; one name per definition. The one exception
  is the stable API (`docs/guide/integrating.md`): its old name stays one release, warning, then goes.
- **Be explicit:** fix a caller rather than loosen a contract check.
- **Python first, then C++:** write the stage in Python with a block-cut test; move hot loops to `csrc/` behind
  `rpps._accel.native`, keeping the Python loop as the reference the tests compare against. A primitive's kernel goes
  behind a `rpps.math`/`rpps.signal` wrapper that stages share; one stage's loop stays in that stage until a second caller.
- **Decoders return status, never keep it** (`decode(x, status=True)`), so one loaded code serves many threads.
- **Read the standard** when a decode does not work, instead of guessing.
- **Statistical tests** assert with a margin measured over many seeds.
- **Docs live with what they describe:** concepts in the package/module docstring, usage in a guide that embeds an
  example script. Python blocks in README.md and `docs/` run in the tests (mark untestable ones
  `# docs: skip`). User-facing changes go in CHANGELOG.md under Unreleased; renames and removals under **Breaking**.
- Branch from `dev`; `main` is released.
