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

    mod = rp.mod.load("QPSK")
    ecc = rp.coding.Hamming(7, 4)
    scr = rp.scram.load("fdt", "v35")

    msg = rp.Data(b"Hello World!")
    syms = msg * scr * ecc * mod          # scramble, encode, modulate
    out = syms / mod / ecc / scr          # demodulate, decode, descramble

    # a receiver is stages added together; blocks flow through it with state carried along
    rx = rp.filters.low_pass(401, 3e6, 100e6) + rp.sample.down.Decimate(8) + rp.sync.Acquire()
    for out in rp.serial.File("ci32t", "capture.cs16", Fs=100e6, cf=1e9)(16384) + rx:
        print(out.Fs, out.sps)

## Where things are

| Module | What it is |
| --- | --- |
| `rpps.base` | `rpps.Data`, `rpps.Pipeline`, `rpps.Pipe`, `rpps.Stage` and `rpps.Stream`: the operator machinery |
| `rpps.serial` | Signal sources and sinks: files, sockets and radios (through pysdrlib) |
| `rpps.filters`, `rpps.sample` | FIR/IIR filters, pulse shaping, resampling |
| `rpps.sync` | AGC, PLLs, Costas, frequency / phase / timing recovery |
| `rpps.equalize`, `rpps.demod`, `rpps.channel` | Equalizers, analog demodulation, channel impairments to test against |
| `rpps.mod` | Modulation: PSK, QAM, APSK, ASK, FSK, OFDM |
| `rpps.coding`, `rpps.chain` | FEC encoders / decoders (block, convolutional, LDPC, turbo...) and whole-standard chains |
| `rpps.scram`, `rpps.encap` | Scramblers, CRC and framing |
| `rpps.dsp`, `rpps.analysis` | Spectra, rate estimation and blind analysis of unknown signals |
| `rpps.math`, `rpps.utils` | Finite fields, GF(2) algebra and other helpers |
| `rpps.viz` | Real-time and over-time plots (`pip install rpps[viz]`) |

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
from .base.types import pproc, dtype
from .base import data
from .base.data import Data
from .base import Pipeline, Pipe, Stage, Stream

from . import base
from . import math
from . import utils

## Import implementations
from . import filters
from . import dsp
from . import sample
from . import sync
from . import analysis

from . import scram
from . import coding
from . import encap
from . import chain
from . import channel
from . import equalize
from . import demod
from .chain import Chain
from . import mod

from . import serial


def __getattr__(name):
    # Plotting pulls in matplotlib, so it's loaded on first use (rp.viz)
    if name == "viz":
        import importlib
        return importlib.import_module(".viz", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
