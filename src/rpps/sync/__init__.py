"""Sync: estimate something unknown about the data and correct or align it, at whatever representation it can be separated

Its estimates go to `stream.est[name]`, never into the data. **Block estimation and block-rate tracking are first
class; per-sample loops are second class.**

## First class: block estimation, tracked across windows

Estimate from a window with block operations (FFTs, correlation, least squares), then a slow loop run **once per window**
(`track`: critically damped second order, wide while it pulls in, coasting through gated windows) carries each parameter
and its rate across windows and gates outliers; the correction is evaluated per item.

| Estimates | Operator | Profile method |
|---|---|---|
| symbol timing | `time.OerderMeyr`: square-law, cubic interpolation; follows a clock 200 ppm off | `timing: oerder_meyr` (default) |
| symbol timing, the matched filter fused in | `time.MatTrack`: the matched filter's energy maximum (MatRRC), read only at the instants; on a GPU too | `timing: mat` |
| carrier frequency and phase | `phase.MPower`: M-th power of the phase, FFT acquisition | `carrier: mpower` (default) |
| channel | `equalize.BlockLS`: least squares per window against decisions | `equalize: ls` (default) |
| OFDM timing and frequency | `CPSync`: from the cyclic prefix | |
| carrier phase per frame | `PilotPhase` (pilot blocks), `phase.FFCarrier` (Viterbi & Viterbi) | `carrier: pilots`, `carrier: ff` |
| frame starts | `FrameSync` (a preamble and the best word of a header code; or named alternatives, Zadoff-Chu sets, a word mid-burst, real levels, what the match says: word, scale, cfo, esn0, modulation), `Delimiter` (a word among bits) | `frame`, `delimit` |
| alignment on bits | `Align` (markers, patterns, polarity), `Puncture` (puncture phase) | `align`, `puncture` |
| modulation, rate, offset | `Acquire` (`detect` once), `Lock` (a snapshot: `Lock.find`, `lock.update`; contiguous blocks joined: `lock.follow`); code phase and Doppler: `acquire_code` | |
| channel code, bit map, alignment | `AcquireFEC` | |
| bursts | `Burst` splits a stream into bursts, `BurstTx` places them; `Bursts` on many carriers, placed in a burst time plan (rpps.plan) whose origin it acquires and tracks, `PlanTx` places them by one; `Preamble` finds each one's preamble (`attach_tx()`: its `PreambleTx`) | `burst`, `preamble` |
| a pilot tone | `ToneLock` (FM stereo's 19 kHz, ATSC's pilot) | |

Rules for feedforward estimators:

1. **Windows sit at absolute stream positions** (`[k*W, k*W + L)`, history carried between blocks), so output does not
   depend on how the stream is cut.
2. **Drifting parameters are tracked across windows**, not held per window (a free-running SDR is 10-20 ppm off and
   wanders), and the correction is evaluated per sample.
3. **Windows are set in time, not symbols**: oscillator phase wanders in time (cycle slips at 8 ms windows, none at 2 ms).
4. **Latency is declared** in `stream.est[name]["latency"]` (seconds), not added to `meta.delay`, the group delay.

A profile's sync steps name **what** is estimated (`carrier`, `timing`, `equalize`, `frame`) and get the first-class
method; a loop is chosen by name in `defaults`: `{"timing": {"method": "gardner", "alpha": 0.3}}`. A new estimator is
registered with `rp.profile.steps.register_sync_method`.

## Second class: per-sample loops

For signals and conditions block estimation does not suit: `time.MM` (Mueller & Muller, Gardner), `time.TLL` (a timing
locked loop around a detector from `time.ted`: `ted="zx"` zero crossing, `"el"` early-late), `Costas`,
`equalize.LMS`, `equalize.CMA`. Profiles reach them by name (`timing: mm`, `timing: {"method": "tll", "ted": "el"}`,
`carrier: costas`, `equalize: lms`). Their loop filter is `pll.PILoop` (gains from a bandwidth: `pll.loop_gains`).
`loops` holds the loops whose first-class stage replaced them: `loops.AGC` (`agc: loop`), `loops.PeakAGC`,
`loops.Squelch`.

## Known, measured, detected

- `time.Known`: timing that is given, not estimated (a loopback, a source whose timing is known); fuses the matched
  filter before it (`timing: known`).
- On a GPU (`rpps._accel.device()`, CUDA or ROCm/HIP; numpy without one), by `"device": true` in a step's sync settings:
  `time.MatTrack` (`timing: mat`: the held samples filtered and read at the instants on the device, windows measured a
  group at a time) and `FrameSync` (`frame`: the preamble metric, and the header words ranked in one matrix product). Both
  agree with their numpy paths to single-precision rounding: the same frames, the same symbols to 1e-4.
- `detect`, `Detection`, `rate.find_baud`: the feed-forward estimates `Acquire` runs (rate, timing, modulation, offset,
  phase); blind analysis uses them too.
- `quality`: Es/N0, Eb/N0 and EVM of locked symbols, once (`measure`), over a window (`Track`) or on the stream
  (`Monitor`, `{"sync": "quality"}`); each symbol's error (`errors`) and an analyser's error summary (`summary`: EVM, MER,
  magnitude and phase error, IQ offset, gain imbalance, quadrature error).

**`AcquireFEC`** does for the code what `Acquire` does for the modulation: on timed, carrier-corrected symbols it finds
the configured code, bit map and alignment (`analysis.identify`: convolutional codes, RS behind a Forney interleaver,
HDLC, block codes), names the profile that decodes it, then tracks by re-encoding once per window and putting carrier
turns and symbol slips right; `est["fec"].rx()` decodes. With `blind=True` the blind tools (`rpps.analysis`) run when no
definition fits; what they find is a `Guess`, a definition that decodes once saved.
"""
from .. import _config


from .costas import Costas
from .acquire import Acquire, Detection, Lock, detect, track_timing
from .time.mat import best_tau, matrix_mf, centres_in
from . import rate, quality
from .rate import find_baud, find_rate
from .acquire_fec import AcquireFEC
from .align import Align
from .puncture import Puncture

from . import phase
from . import time

from . import agc
from .agc import AGC, PeakAGC
from . import pll
from . import equalize
from .squelch import Squelch
from . import loops
from .code_acquire import acquire_code
from .pilots import PilotEqualizer, PilotPhase
from .frame import FrameSync, Delimiter, word_errors
from .ofdm import CPSync
from .burst import Burst, BurstTx
from .tone import ToneLock
from .preamble import Preamble, PreambleTx
from .plan import Bursts, PlanTx

from ..base.rpps import tag_stages as _tag_stages
from ..base.types import Category as _Category
_tag_stages("rpps.sync", _Category.SYNC)
