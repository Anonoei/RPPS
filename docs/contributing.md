# Contributing

How RPPS is built and how to add to it. The design (the model, the rule that decides where a function goes, the
categories' contracts, the streaming rules) is [`rpps.base`](api/base.md); read it before adding an operator.

## Where documentation goes

| Kind | Home |
|---|---|
| What a package, module or operator is and why (design, rules, contracts, method, references) | its docstring: the API pages render it |
| How to use RPPS for a task (recommended ways, end to end) | a guide in `docs/guide/`, embedding a script from `examples/` with `--8<-- "examples/<name>.py"` |
| A runnable, self-checking demonstration | `examples/<name>.py`, run by `tests/unit/policy/test_examples.py` |
| What agents need first | `AGENTS.md`; `llms.txt` / `llms-full.txt` are generated from it and the docstrings |
| Definition file formats | `docs/guide/definitions.md` |

Python blocks in README.md and `docs/` run in `tests/unit/policy/test_docs.py` unless they hold `# docs: skip`. `python tools/builder.py
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
`dependencies` (numpy, pysdrlib), extras `viz` (matplotlib), `test` (pytest, pytest-xdist, coverage), `oracle` (scipy), `docs` (zensical, mkdocstrings-python),
`dev` (all of them plus twine, build, bumpver). It also holds `[tool.cibuildwheel]`, `[tool.bumpver]` and the pytest markers.

Branch from `dev`, merge back into it; `main` is released. Scratch scripts are not committed; worked examples that reuse
RPPS and may be upstreamed go in `projects/`.


## Repository layout

```
src/rpps/            the package: one sub-package per category (rpps, Where things are)
csrc/rpps_native/    C++ kernels, built as rpps._native (optional)
config/              JSON definitions, copied into the wheel as rpps/config
tests/unit/          the test suite, a folder per package
tests/support/       helpers the tests share; tests/vectors/ reference vectors
tests/run_wheel_tests.py  the unit tests against an installed wheel (cibuildwheel)
examples/            one script per recommended way of doing a thing, each checking itself
docs/                the guides (markdown; docs/api/ and the llms files are generated)
projects/            worked examples on real signals (FM, ADS-B, ATSC, LTE, P25 ...), each with a README
data/                captures (git-ignored)
tools/               builder.py (the local build helper), gen_docs.py (API pages and llms files for the site),
                     benchmarks run by hand (bench_stages.py, parallel_bench.py, resample_bench.py, effort_bench.py,
                     modclass_bench.py, blind_bench.py, ident_bench.py, tpc_ber.py); each takes --save, --update-baseline and --check (tools/_bench.py),
                     keeping this machine's results in the git-ignored .benchmarks/<bench>/
setup.py             builds the native module and copies config/
```

`python -m rpps` has `ls`, `show <ref>`, `steps`, `rx`, `tx`, `survey`, `profile`, `solve` and `fmt`. In code,
`rp.load("<resource>/<name>", **params)` loads any definition.


## Adding things

- **A definition** (constellation, code, scrambler, pulse, CRC, sequence, header, table, profile): a JSON file in
  `config/<resource>/` (the folder says what it is) with `type`, `comment` and `ref`; `python -m rpps ls <resource>` to see it,
  `python -m rpps fmt --check` for style.
- **An implementation** (a code type, a sync method): put it in its category's package (`rpps.base`, The rule); subclass
  `Pipe`, `Stage` or `BitStage` with `CATEGORY`, `takes` and `SCOPE`; register its loader and its profile step
  (`rp.profile.steps.register_step`, `register_sync_method`), or name any `Stage` by module path in a profile
  (`{"signal": "class", "class": "mypkg.Gain", "args": {...}}`). A code type subclasses `code.types._code._code` and defines
  `_decode(x, ...) -> (output, errors per codeword or None)`, keeping nothing of the call on itself. Write Python first
  with a block-cut test; move hot loops to C++ after.
- **A sync estimator:** block operations over windows at absolute positions, tracked by `sync.track`. A per-sample loop is
  second class, chosen by name in `defaults`.
- **A signal:** a profile, which should need no new operator code ([Profiles](guide/profiles.md)).
- **A source:** subclass `Source`, say what representation and sync it gives, set its rates and each block's position.
- **Branches:** `data + rp.Tee(a=..., b=...)` runs each branch in its own stream (FM stereo L+R and L-R); profiles stay one chain.


## Native kernels

`rpps._native` (C++17, pybind11, `csrc/rpps_native/*.cpp`) holds the per-sample loops and decoders (AGC, Costas, timing,
equalizers, IIR sections, the one-pole loops of the `loops` packages, Viterbi, BCJR, LDPC, polar list decoding, RS, BCH, demapping, resampling, FIR decimation, carrier and
frame sync), bit-level work on hard bits (`bits.cpp`: the Hamming distance to a word at every position, bit destuffing;
`rpps.math.bits.distance`, `find`, `unstuff`) and the blind
tools' slow loops (`analysis.cpp`). Without a compiler the build warns and RPPS runs its Python loops.

**The Python loop is the reference.** Every kernel's Python version stays in the stage that calls it, and tests compare the
two (`tests/unit/native/`, `tests/unit/parallel/`): a kernel changes speed, never results beyond rounding.
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

**Where a kernel's Python side lives.** A kernel that is a primitive on plain arrays (bits, correlation, interpolation,
GF(2) elimination) sits behind a wrapper in `rpps.math` or `rpps.signal` that holds the Python reference, and stages call
the wrapper (`rpps.math.bits.distance` serves FrameSync, HDLC and blind framing). A kernel for one stage's loop stays
behind that stage until a second caller appears, then moves down to a wrapper. Don't generalize a kernel ahead of
its second caller. Portable bit helpers (`ctz64`, `popcount64`) are in `common.hpp`.

Pitfalls found the hard way:
- **`long long` loop and tap indices.** CPython builds with `-fno-strict-overflow`, so `int` indices are not widened: 3x
  slower inside the module.
- **`target_clones("avx2", "default")` only**, behind `#if` for GCC (MSVC has neither it nor `__builtin_*`; `common.hpp` has
  portable helpers such as `ctz64`). `"arch=haswell"` made the 4-accumulator dot product 40% slower (FMA lengthens the chain).
- **Size `parallel_for` chunks by work** (about 128k multiply-adds), not items; the pool is persistent and reset after a fork.
- **Keep BLAS products out of per-block paths:** OpenBLAS spins between calls and slows everything around it
  (`OPENBLAS_NUM_THREADS=1` shows it).
- **Benchmark best-of-N in one process** (`tools/bench_stages.py`, `parallel_bench.py`, `resample_bench.py`).

## Device kernels (GPU)

A parallel region runs on a GPU through CuPy (`rpps._accel.device()`, `rpps.parallel`), built for CUDA or for ROCm/HIP with
one API; nothing in RPPS names a vendor. CuPy is not a dependency: keep it in its own environment.

```
pyenv virtualenv 3.12.11 rpps-gpu && pyenv shell rpps-gpu
pip install -e ".[dev]"
pip install cupy-cuda12x                                        # NVIDIA: a wheel
CUPY_INSTALL_USE_HIP=1 ROCM_HOME=/opt/rocm HCC_AMDGPU_TARGET=gfx1152 pip install --no-binary cupy cupy   # AMD: from source
```

The ROCm build needs hipcub, rocthrust, hiprand/rocrand and hipfft/rocfft installed; `HCC_AMDGPU_TARGET` is the GPU's
`gfx` name (`rocminfo`). With CuPy present, the tests in `tests/unit/parallel/` run every kernel on the GPU as
well as on the pretend device (`tests/support/fakedevice.py`, which CI uses); `RPPS_DEVICE=none` leaves the GPU out.

- **Array kernels** use only what numpy and cupy share (`xp`). **Device kernels** of their own (`LdpcDecode`,
  `BchDecode`, `Demap`) are portable CUDA C that NVRTC and hiprtc both compile (`rpps.base.kernel.device_functions`): no
  warp intrinsics or warp-width assumptions, no vendor headers. On numpy, or a module without kernels (fakedevice), they
  run the host decoder, which stays the reference.
- They compute in float32 (consumer GPUs run float64 at a fraction of the rate): tests compare decisions where the host
  converges, and frame failures at the waterfall within a margin.
- Measure with `python tools/device_bench.py` (per kernel, host against device, by batch size) and
  `python tools/parallel_bench.py 64 4 --device --batch 16` (the DVB-S2 region end to end), on an idle machine: the host
  side uses every core, so a busy one flatters the GPU.

Pitfalls found the hard way:
- **The batch is the fastest axis** in a device kernel's arrays (LDPC's `total[N, B]`, `r[E, B]`): neighbouring threads
  then touch neighbouring words. Codeword-major, the same kernel was 11-13x slower once the batch outgrew the cache.
- **Compact what is done.** A wavefront runs while any of its lanes works: LDPC moves the codewords still decoding into
  a narrower array as others converge, else a few stragglers cost a whole batch every iteration.
- **No boolean indexing or `bool()` in a device path:** each is a search and a wait for the device. Index with integer
  constants (`Gather`), and read back once per loop at most.
- **float64 is slow on consumer GPUs** (the generic array demapper was 3x slower than the C++ one until it had a float32
  kernel of its own).
- **An integrated GPU shares the CPU's power:** with every core busy the Radeon 860M held 800-900 MHz of its 3000
  (158 GFLOP/s SGEMM, 16 GB/s), so a GPU path that wins on an idle machine can tie with the CPU on a loaded one, and a
  CPU front competes with a GPU region inside one receiver. Check `pp_dpm_sclk` (or `rocm-smi --showclocks`) and the
  load before trusting a number.
- **A front stage that tracks window by window waits on the device every window.** `OerderMeyr` and `MPower` with
  `device=True` measure each window with one reduction and read the sums back before the tracker can place the next
  window: on the 860M, with 1 ms windows, the DVB-S2 receiver was 1.3x slower with both on the device than with the C++
  host paths. `MatTrack` (`timing: mat`) measures a group of windows a launch and wins (16%). Batch a group of windows
  before moving a front stage to a device.
- **ROCm packages out of step:** a hipcub newer than rocprim fails CuPy's build (`segmented_reduce_config_selector`). Build
  against the hipcub release that matches rocprim (`HIPCUB_VERSION` = `ROCPRIM_VERSION` in their `*_version.hpp`), for
  example by pointing `ROCM_HOME` at a copy of `/opt/rocm` made of links with that hipcub's headers in `include/hipcub`.


## Testing

Where a test goes (`policy/test_test_layout.py` holds the folder rules):

| Test of | Folder under `tests/unit/` |
|---|---|
| a package, `rpps.<name>` (and its sub-packages, as `analysis/spectrum/`) | `<name>/` |
| definitions, parameters, the command line | `config/` |
| a whole standard, received or built from its definitions | `standards/` |
| a C++ kernel against its Python loop | `native/` |
| a rule of the repository: layering, imports, contracts, docs, examples | `policy/` |

File names are unique across the folders, and say the subject.

```
python -m pytest -n auto -m "not slow"                       # the fast tests (CI runs this)
RPPS_NO_NATIVE=1 python -m pytest -n auto -m "not slow and not kernels"   # the Python loops
python -m pytest -m oracle                                   # the scipy comparisons (needs .[oracle])
python -m pytest tests/unit/sync                             # one package
python -m pytest tests/unit/signal/test_remez.py             # one file
python -m pytest -m native                                   # the C++ kernels against the Python reference
coverage run -m pytest -n auto && coverage combine && coverage report
```

`-n auto` runs on half the cores (every core where `CI` is set), and each worker keeps BLAS and RPPS's own pools to
one thread (`tests/unit/conftest.py`), so the machine stays usable; `RPPS_TEST_WORKERS=4` chooses the workers.
An idle worker takes tests queued on a busy one (`--dist=worksteal`, in `pyproject.toml`): a few tests take most of the
time.

| Marker | Meaning |
|---|---|
| `slow` | more than a few seconds even with the kernels; not run in CI |
| `kernels` | over a minute without the kernels; left out of the no-kernel run |
| `native` | everything in `tests/unit/native/` (set by its `conftest.py`); skipped without the kernels |
| `oracle` | compares against scipy through `pytest.importorskip`; skipped without it |

- **RPPS must not need scipy**; nothing under `src/` may import it (`policy/test_numpy_only.py`).
- `tests/unit/conftest.py` sets the Agg backend and gives the fixtures `rng` (seeded) and `config_layer` (a user config
  directory for one test). A test that leaves a config layer or the kernel switch changed fails.
- `tests/support/` holds what several files share, imported as `from support.signals import awgn`: signals, block
  cuts, the kernel switch (`support.native`). No test file imports another.
- `policy/test_contracts.py` runs every definition against its category's contract, so new definitions are covered automatically.
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
- **No aliases:** a rename or move drops the old name and fixes the callers. The one exception is the stable API
  ([Building on RPPS](guide/integrating.md)): its old name stays one release, warning, then goes.
- **Be explicit:** fix a caller rather than loosen a contract check.
- **Category by what it takes and gives** (`rpps.base`, The rule).
- **Read the standard** when a decode does not work, instead of guessing.
- **Examples live in `examples/` and `projects/`**, scratch stays uncommitted; reuse RPPS so useful parts can move
  into the package.
