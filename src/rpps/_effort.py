"""How hard the receiver works: speed against recovered SNR

    rp.set_effort("max")                 # from now on
    with rp.effort("fast"):              # for a block of code
        out = rx(block)
    RPPS_EFFORT=max python ...           # for a process

Levels (looked up when a stage or decoder runs, so a level can change mid-stream):

    fast       the cheapest work that still decodes at good SNR: max-log demapping, half the decoder iterations, half
               the polar list, one Chase bit less, cubic timing interpolation
    balanced   the default: what each definition says (its iterations, list, Chase bits, demapper, min-sum), timing
               interpolation that keeps the interpolation error under the pulse's own truncation at 3 samples per
               symbol and more
    max        the most SNR at more cost: decoders iterate further (up to 4x; each codeword still stops when it
               converges, so the cost is mostly near the threshold), LDPC by sum-product, 4x the polar list, one Chase
               bit more, log-MAP turbo decoding, a timing interpolator at the pulse's own floor at 2 samples per symbol

A definition's settings stay the base: a level scales them (an LDPC definition with 50 iterations runs 50 balanced, 200
max), and a value set on an operator is that operator's base. `rpps rx --effort` sets it from the shell.

| Setting | fast | max |
|---|---|---|
| LDPC / turbo / TPC iterations (`iters`) | x0.5 | x4, stopping early |
| LDPC check update (`check`) | min-sum | sum-product |
| Polar list (`list`) | x0.5 | x4 (successive cancellation lists 4) |
| TPC Chase-II bits (`chase`) | -1 | +1 |
| Max-log or exact (`maxlog`: demapper, BCJR, turbo) | max-log | exact, no extrinsic scaling |
| Timing interpolation (`interp_taps`) | cubic | 12-tap windowed sinc (balanced: 8 taps below 4 sps) |

Measured (`tests/effort_bench.py`, FER 1e-1, one thread; Mbit/s at the threshold, where max iterates most):

| Chain | Es/N0 dB fast / balanced / max | Mbit/s fast / balanced / max |
|---|---|---|
| LDPC DVB-S2 64800 1/2, QPSK | 1.08 / 0.94 / 0.75 | 2.2 / 1.2 / 0.96 |
| LDPC DVB-S2 16200 3/5, 8PSK | 6.16 / 6.09 / 5.62 | 7.7 / 2.7 / 0.79 |
| Turbo LTE K=6144, QPSK | -0.84 / -1.31 / -1.62 | 1.6 / 0.85 / 0.10 |
| Turbo LTE K=40, QPSK | -0.34 / -0.41 / -0.49 | 0.47 / 0.24 / 0.073 |
| Polar NR DCI (K 63, E 432), QPSK | -4.97 / -5.25 / -5.67 | 0.22 / 0.12 / 0.032 |
| Polar NR PBCH, QPSK | -8.50 / -8.85 / -9.32 | 0.19 / 0.097 / 0.029 |
| Polar (256,128) + CRC-16, QPSK | 1.29 / 1.13 / 0.90 | 0.92 / 0.38 / 0.098 |
| TPC (64,57)^2, BPSK | 2.81 / 2.16 / 1.69 | 0.36 / 0.13 / 0.032 |
| Conv K7 1/2 (Viterbi), QPSK | 3.61 / 3.61 / 3.61 | 7.9 / 5.6 / 5.6 |

Timing at 2 sps: EVM floor -28.0 / -48.5 / -55.4 dB (pulse truncation -56) at the same speed.

Other decoding behaviour worth knowing:

- LDPC is layered (checks coloured once per code), about half the iterations of flooding: normalized min-sum, or
  sum-product at max (phi tabulated, 1.7x min-sum's time an iteration, fewer iterations).
- Soft values go as far as something uses them: additive descrambling flips LLR signs, Forney deinterleaving moves LLRs,
  BCJR hands on a-posteriori LLRs (RS uses them for erasures, keeping two checks spare), differential decoding is the
  product detector `y conj(y[n-1])`.
- A decoder returns its status, it never keeps it (`decode(x, status=True)`). A code step puts the errors on its output
  (`data.errors`, not carried further) and sets `data.failed` from them (the last code with a verdict decides); later
  steps carry `failed`, and a link header drops a frame with any failed item whatever its CRC says. The parallel region
  drops such rows.
- Under 4 sps Oerder & Meyr takes its square-law line at twice the rate (at 2 sps it aliased onto its mirror).
"""
import contextlib
import os

LEVELS = ("fast", "balanced", "max")

# What each level sets: {setting: {level: value}}
SETTINGS = {
    "iters": {"fast": 0.5, "balanced": 1.0, "max": 4.0},          # factor on a decoder's iteration cap (LDPC, turbo, TPC)
    "list": {"fast": 0.5, "balanced": 1.0, "max": 4.0},           # factor on a polar list decoder's paths (1: successive cancellation)
    "chase": {"fast": -1, "balanced": 0, "max": 1},                 # added to a turbo product code's Chase-II test bits (2^bits patterns)
    "check": {"fast": None, "balanced": None, "max": "sumproduct"}, # LDPC layered check update: the definition's (min-sum), or sum-product
    "maxlog": {"fast": True, "balanced": None, "max": False},      # demapping and turbo: max-log (True), exact (False), the definition's (None)
    "interp_taps": {"fast": 4, "balanced": None, "max": 12},        # timing interpolation: 4 = cubic, else windowed sinc taps; None: by sps
}

def _check(level):
    if level not in LEVELS:
        raise ValueError(f"effort is one of {LEVELS}, not {level!r}")
    return level

_level = _check(os.environ.get("RPPS_EFFORT", "balanced"))

def level():
    """The effort level in force"""
    return _level

def set_effort(level):
    """Set the effort level from now on: "fast", "balanced" or "max". Returns the level it replaces"""
    global _level
    old, _level = _level, _check(level)
    return old

@contextlib.contextmanager
def effort(level):
    """The effort level for a `with` block"""
    old = set_effort(level)
    try:
        yield
    finally:
        set_effort(old)

def get(name):
    """A setting at the level in force"""
    return SETTINGS[name][_level]

def interp_taps(sps):
    """Taps of the timing interpolation at the level in force for `sps` samples per symbol: balanced keeps cubic (4) from 4
    samples a symbol, where its error (-49 dB) is under the pulse's own truncation, and takes 8 under it (cubic: -26 dB at 2)"""
    v = SETTINGS["interp_taps"][_level]
    return v if v is not None else (4 if sps >= 4 else 8)

def scale(name, base):
    """A count at the level in force ("iters", "list"): its own (`base`) scaled, at least 1"""
    return max(1, int(round(base*SETTINGS[name][_level])))

def shift(name, base):
    """A count at the level in force ("chase"): its own (`base`) plus the level's step, at least 1"""
    return max(1, int(base) + SETTINGS[name][_level])

def choice(name, base):
    """A choice at the level in force ("maxlog", "check"): the level's, else the operator's own (`base`)"""
    v = SETTINGS[name][_level]
    return base if v is None else v
