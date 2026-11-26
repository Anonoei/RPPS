"""RPPS: RF Parameter Processor Suite, a generic signal processor/generator library

Everything is driven by operators: `*` encodes / modulates / scrambles, `/` undoes it, `+` runs a DSP stage, `>>` writes
to a sink. The data carries its sample rate and the state of every stage from block to block. The design (the model,
categories, contracts, streaming rules) is in `rpps.base`; how to use it is in the guides of the documentation site
(https://anonoei.github.io/RPPS/).

## Where things are

Packages follow the categories of `rpps.base`; data goes samples -> symbols -> channel bits -> data bits -> packets.

| Package | What is in it |
|---|---|
| `rpps.source` | files and their formats, WAV, SigMF, sockets, radios through pysdrlib (imported only when used); sinks |
| `rpps.signal` | `filters` (FIR, IIR, windows, pulses, fractional delay, polyphase channelizer), `sample` (resampling), `channelize`, `DCBlock`, `Deemphasis`, `Conjugate`, `channel` (the simulated medium); `loops` (the per-sample DCBlock, Deemphasis) |
| `rpps.sync` | first class: `time.OerderMeyr`, `phase.MPower`, `phase.DDCarrier`, `equalize.BlockLS`, `CPSync`, tracked by `track`; `FrameSync`, `Delimiter`, `PilotPhase`, `PilotEqualizer`, `Align`, `Puncture`, `Acquire`, `Lock`, `AcquireFEC`, `acquire_code`, `Squelch`, `AGC`, `PeakAGC`, `Burst`, `BurstTx`, `Bursts` (many carriers, a burst time plan), `PlanTx`, `Preamble`, `ToneLock`. Second class: `loops` (`MM` (Gardner), `TLL`, `Costas`, `AGC`, `PeakAGC`, `Squelch`, `pll`, `LMS`, `CMA`) |
| `rpps.wave` | `Shape`/`Matched`, `Stagger`, `FM`/`FMDemod`, `OFDMTx`/`OFDMRx`, `AMDemod`, `SSBDemod`, `demod`; `loops` (the per-sample AMDemod) |
| `rpps.symbol` | `Constellation` (PSK, QAM, APSK, ASK), `Differential`, `Spread`/`Despread`, `Carriers`, `Header`, `Pilots`, `PhaseScramble` |
| `rpps.code` | FEC, interleaving, rate matching (`code.types`), scramblers (`code.scram`), `Line`, `Marker`, `Uncoded` |
| `rpps.link` | CRCs, HDLC, `Header` (fields, CRC, payload length) |
| `rpps.grid` | resource grids of OFDM systems: LTE and 5G NR cell search, grids, physical channels |
| `rpps.profile` | a whole signal from one config file, both directions: loading, steps, header-driven control, runners |
| `rpps.parallel` | a profile's frame-parallel region: executors, scheduler, batch kernels |
| `rpps.sequence` | recurrences, Gold sequences, chip codes, words: what scramblers, spreaders and searches share |
| `rpps.plan` | burst time plans (TDMA, MF-TDMA, hopping, trains): superframes, frames, timeslots, burst types, assignments; from a definition, a forward link's tables (DVB-RCS2) or the bursts themselves |
| `rpps.analysis` | outside the data path: spectra, rates, surveys, blind analysis of modulation and FEC; can write definitions |
| `rpps.base` | `Data`, `Pipe`, `Stage`, `BitStage`, `Pipeline`, `Stream`, `Tee`, type tags, control, kernels |
| `rpps.math`, `rpps.utils`, `rpps.viz` | finite fields and GF(2) algebra; helpers; plots (`pip install rpps[viz]`) |

Imports run down this list: nothing in the data path imports `analysis`, `profile` or `parallel` (`test_layering.py`;
the one exception is `AcquireFEC`, which imports `analysis.identify` inside the call). `import rpps` loads numpy and
RPPS only: no scipy, matplotlib or pysdrlib until something asks for them.

Constellations, pulses, filters, sequences, scramblers, codes, headers, tables and profiles are definitions that ship
with the package (`python -m rpps ls` lists them; `rp.load("<resource>/<name>")` loads any); add your own with
`rpps.add_config_path`. How hard decoders and timing work is `rpps.set_effort` (levels and measured gains in
`rpps._effort`). The per-sample loops and decoders run in C++ (`rpps._native`) with a Python fallback.
"""

__version__ = "0.6.0"
__author__ = "Anonoei <to+dev@an0.us>"

from . import logger
from . import errors
from . import _config
from ._config import add_config_path, use_shipped, select_tags, domains, tags, tagged
from ._config import load_ref as load
from ._effort import set_effort, effort

# Import helpers/globals
from .base.types import Category, Rep, dtype
from .base import data
from .base.data import Data
from .base import Pipeline, Pipe, Stage, BitStage, Stream, Tee

from . import base
from . import math
from . import utils

## Import implementations
from . import signal
from . import sync
from . import analysis

from . import code
from . import link
from . import sequence
from . import plan
from . import wave
from . import profile
from . import parallel
from . import symbol

from . import source


def __getattr__(name):
    # Plotting pulls in matplotlib, so it's loaded on first use (rp.viz)
    if name == "viz":
        import importlib
        return importlib.import_module(".viz", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
