# Contributing

How RPPS is built and how to add to it. The design (the model, the rule that decides where a function goes, the
categories' contracts, the streaming rules) is [`rpps.base`](api/base.md); read it before adding an operator.

## Where documentation goes

| Kind | Home |
|---|---|
| What a package, module or operator is and why (design, rules, contracts, method, references) | its docstring: the API pages render it |
| How to use RPPS for a task (recommended ways, end to end) | a guide in `docs/guide/`, embedding a script from `examples/` with `--8<-- "examples/<name>.py"` |
| A runnable, self-checking demonstration | `examples/<name>.py`, run by `tests/unit/test_examples.py` |
| What agents need first | `AGENTS.md`; `llms.txt` / `llms-full.txt` are generated from it and the docstrings |
| Definition file formats | `docs/guide/definitions.md` |

Python blocks in README.md and `docs/` run in `tests/unit/test_docs.py` unless they hold `# docs: skip`. `python tools/builder.py
-d` builds the site with `--strict`, so a broken snippet path or docstring cross-reference fails the build.

## Setup and workflow

```
git clone https://github.com/Anonoei/RPPS && cd RPPS
python tools/builder.py -s        # pip install -e ".[dev]"
python tools/builder.py -c        # build the C++ kernels in place (C++17 compiler)
python tools/builder.py -t        # pytest
```

`tools/builder.py -h` lists the rest: `-d` docs (API pages, llms.txt, then `zensical build` into `site/`; `zensical serve` to preview), `-b` sdist and wheel (then `twine check`), `-l` local install, `-v M|m|p` version
bump and `-u` upload (both dry runs without `-r`; upload goes to TestPyPI without it).

`pyproject.toml` is the only place that names a dependency, CI included: `[build-system]` (setuptools, numpy, pybind11),
`dependencies` (numpy, pysdrlib), extras `viz` (matplotlib), `test` (pytest, pytest-xdist), `oracle` (scipy), `docs` (zensical, mkdocstrings-python),
`dev` (all of them plus twine, build, bumpver). It also holds `[tool.cibuildwheel]`, `[tool.bumpver]` and the pytest markers.

Branch from `dev`, merge back into it; `main` is released. Scratch scripts are not committed; worked examples that reuse
RPPS and may be upstreamed go in `projects/`.


## Repository layout

```
src/rpps/            the package: one sub-package per category (rpps, Where things are)
csrc/rpps_native/    C++ kernels, built as rpps._native (optional)
config/              JSON definitions, copied into the wheel as rpps/config
tests/unit/          the test suite
tests/*.py           benchmarks (bench_stages.py, parallel_bench.py, tpc_ber.py)
examples/            one script per recommended way of doing a thing, each checking itself
docs/                the guides (markdown; docs/api/ and the llms files are generated)
projects/            worked examples on real signals (FM, ADS-B, ATSC, LTE, P25 ...), each with a README
data/                captures (git-ignored)
tools/               builder.py (the local build helper), gen_docs.py (API pages and llms files for the site)
setup.py             builds the native module and copies config/
```

`python -m rpps` has `ls`, `show <ref>`, `steps`, `rx`, `tx`, `survey`, `profile`, `solve` and `fmt`. In code,
`rp.load("<resource>/<name>", **params)` loads any definition.


## Adding things

- **A definition** (constellation, code, scrambler, pulse, CRC, sequence, header, table, profile): a JSON file in
  `config/<resource>/` with `base`, `type`, `comment` and `ref`; `python -m rpps ls <resource>` to see it,
  `python -m rpps fmt --check` for style.
- **An implementation** (a code type, a sync method): put it in its category's package (`rpps.base`, The rule); subclass
  `Pipe`, `Stage` or `BitStage` with `CATEGORY`, `takes` and `SCOPE`; register its loader and its profile step
  (`rp.profile.steps.register_step`, `register_sync_method`), or name any `Stage` by module path in a profile
  (`{"signal": "stage", "class": "mypkg.Gain", "args": {...}}`). A code type subclasses `code.types._code._code` and defines
  `_decode(x, ...) -> (output, errors per codeword or None)`, keeping nothing of the call on itself. Write Python first
  with a block-cut test; move hot loops to C++ after.
- **A sync estimator:** block operations over windows at absolute positions, tracked by `sync.track`. A per-sample loop is
  second class, chosen by name in `defaults`.
- **A signal:** a profile, which should need no new operator code ([Profiles](guide/profiles.md)).
- **A source:** subclass `Source`, say what representation and sync it gives, set its rates and each block's position.
- **Branches:** `data + rp.Tee(a=..., b=...)` runs each branch in its own stream (FM stereo L+R and L-R); profiles stay one chain.


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


## CI and releases

| Workflow | When | Does |
|---|---|---|
| `tests.yml` | push to `main` / `dev`, pull requests | fast tests on Python 3.12-3.14, Ubuntu and Windows (`-n auto`; Windows 30 min, Ubuntu 20); the oracle tests on 3.12; a no-kernel job; sdist → wheel and the examples in a clean venv; the docs build |
| `deploy-docs.yml` | push to `main` | `python tools/builder.py -d` (Zensical) to GitHub Pages |
| `python-publish.yml` | a GitHub release | cibuildwheel (CPython 3.12-3.14; Linux x86_64 and arm64, Windows; each wheel must carry `rpps._native`), checks the tag against the version, publishes by trusted publishing |

To release: move CHANGELOG.md's "Unreleased" under the new version (every rename or removal under **Breaking**, with
what replaces it); `python tools/builder.py -v p` (or `m`, `M`) previews the bump, `-r` makes it (edits `pyproject.toml` and
`src/rpps/__init__.py`, commits, does not tag); merge to `main`; publish a GitHub release tagged exactly `v<version>`.


## Conventions

- **Operators first:** keep features inside `*`, `/`, `+`, `>>` rather than adding setup calls.
- **No aliases:** a rename or move drops the old name and fixes the callers.
- **Be explicit:** fix a caller rather than loosen a contract check.
- **Category by what it takes and gives** (`rpps.base`, The rule).
- **Read the standard** when a decode does not work, instead of guessing.
- **Examples live in `examples/` and `projects/`**, scratch stays uncommitted; reuse RPPS so useful parts can move
  into the package.
