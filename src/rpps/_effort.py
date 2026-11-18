"""How hard the receiver works: speed against recovered SNR

    rp.set_effort("max")                 # from now on
    with rp.effort("fast"):              # for a block of code
        out = rx(block)
    RPPS_EFFORT=max python ...           # for a process

Levels (looked up when a stage or decoder runs, so a level can change mid-stream):

    fast       the cheapest work that still decodes at good SNR: max-log demapping, fewer decoder iterations, cubic timing
               interpolation
    balanced   the default: what each definition says (its iterations, its demapper), timing interpolation that keeps
               the interpolation error under the pulse's own truncation at 3 samples per symbol and more
    max        the most SNR at more cost: decoders iterate further (LDPC up to 4x; each codeword still stops when it
               converges, so the cost is only near the threshold), log-MAP turbo decoding, a timing interpolator at the
               pulse's own floor at 2 samples per symbol

A definition's settings stay the base: a level scales them (an LDPC definition with 50 iterations runs 50 balanced, 200
max), and a value set on an operator is that operator's base. Measured gains (README.dev.md, Effort): LDPC 0.1-0.15 dB
at max, turbo 0.1-0.2 dB, timing interpolation 0.9 dB at 2 samples per symbol and 20 dB Es/N0.
"""
import contextlib
import os

LEVELS = ("fast", "balanced", "max")

# What each level sets: {setting: {level: value}}
SETTINGS = {
    "iters": {"fast": 0.5, "balanced": 1.0, "max": 4.0},          # factor on a decoder's iteration cap (LDPC, turbo)
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

def iters(base):
    """A decoder's iteration cap at the level in force: its own (`base`) scaled, at least 1"""
    return max(1, int(round(base*SETTINGS["iters"][_level])))

def maxlog(base):
    """Max-log or exact at the level in force: the level's choice, else the operator's own (`base`)"""
    v = SETTINGS["maxlog"][_level]
    return base if v is None else v
