"""
# RPPS

RF Parameter Processor Suite

RPPS is a generic signal processor/generator library.

Github: https://github.com/Anonoei/RPPS

PyPI: https://pypi.org/project/rpps/

## Quick start

Everything is driven by operators: `*` encodes / modulates / scrambles, `/` undoes it, `+` runs a DSP stage.
The data carries its sample rate and the state of every stage, so there is nothing to set up or reset.

    import rpps as rp

    mod = rp.symbol.load("QPSK")
    ecc = rp.code.Hamming(7, 4)
    scr = rp.code.scram.load("fdt", "v35")

    msg = rp.Data(b"Hello World!")
    syms = msg * scr * ecc * mod          # scramble, encode, modulate
    out = syms / mod / ecc / scr          # demodulate, decode, descramble

    # a receiver is stages added together; blocks flow through it with state carried along
    rx = rp.signal.filters.low_pass(401, 3e6, 100e6) + rp.signal.sample.down.Decimate(8) + rp.sync.Acquire()
    for out in rp.source.File("ci32t", "capture.cs16", Fs=100e6, cf=1e9)(16384) + rx:
        print(out.Fs, out.sps)

## Where things are

Packages follow the categories of ARCHITECTURE.md: data goes samples -> symbols -> channel bits -> data bits -> packets.

| Package | Category: what it does |
| --- | --- |
| `rpps.source` | Source: sources and sinks of blocks (files, sockets, radios through pysdrlib) |
| `rpps.signal` | Conditioning, samples to samples: filters, resampling, `channelize`, and `signal.channel`, the simulated medium |
| `rpps.sync` | Sync, at any representation: AGC, carrier, timing, equalizers, alignment on bits (`sync.Align`) |
| `rpps.wave` | Wave transform, symbols <-> samples: stagger (O-QPSK), frequency modulation (FSK, GFSK, GMSK, C4FM), OFDM IFFT/FFT, AM/SSB |
| `rpps.symbol` | Symbol transform, symbols <-> channel bits: constellations (any points from config), differential coding, OFDM carriers, `Spread` |
| `rpps.spread` | Spreading codes (Barker, GPS C/A, 802.15.4), until they become the `sequence` resource |
| `rpps.code` | Code transform, channel bits <-> data bits: FEC, interleaving, rate matching, scrambling (`code.scram`), `Uncoded` |
| `rpps.link` | Link transform, data bits <-> packets: CRC and framing |
| `rpps.analysis` | Outside the data path: spectra, rate estimation, signal survey, blind analysis of modulation and FEC |
| `rpps.base` | `rpps.Data`, `rpps.Pipe`, `rpps.Stage`, `rpps.BitStage`, `rpps.Pipeline`, `rpps.Stream`, and the type tags |
| `rpps.math`, `rpps.utils`, `rpps.viz` | Finite fields and GF(2) algebra, helpers, plots (`pip install rpps[viz]`) |
| `rpps.profile` | Profiles: a whole signal from one config file, receive and transmit (`rp.profile.load`) |

Codes, scramblers and modulations are definitions that ship with the package; add your own with
`rpps.add_config_path`. The per-sample loops and decoders run in C++ (`rpps._native`) with a Python fallback.
"""

__version__ = "0.3.0"
__author__ = "Anonoei <to+dev@an0.us>"

from . import logger
from . import errors
from . import _config
from ._config import add_config_path

# Import helpers/globals
from .base.types import Category, Rep, dtype
from .base import data
from .base.data import Data
from .base import Pipeline, Pipe, Stage, BitStage, Stream

from . import base
from . import math
from . import utils

## Import implementations
from . import signal
from . import sync
from . import analysis

from . import code
from . import link
from . import spread
from . import wave
from . import profile
from . import symbol

from . import source


def __getattr__(name):
    # Plotting pulls in matplotlib, so it's loaded on first use (rp.viz)
    if name == "viz":
        import importlib
        return importlib.import_module(".viz", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
