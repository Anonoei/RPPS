"""Second class: per-sample loops (rpps.sync)

Each updates its estimate once a sample or a symbol, carrying it from one to the next. Block estimation tracked across
windows (rpps.sync's first class) replaces them in the profiles and examples; they stay for signals and conditions it
does not suit, as references, and for comparison. Each runs in the C++ kernels when they are built (rpps._native), with
its Python loop as the reference. Their loop filter is `pll.PILoop`, its gains from a noise bandwidth (`pll.loop_gains`).

| Loop | Estimates | First-class counterpart |
|---|---|---|
| `MM` | symbol timing: Mueller & Muller (`ted="mm"`, decisions) or Gardner (`"gardner"`), interpolated per symbol | `time.OerderMeyr`, `time.MatTrack` |
| `TLL` | symbol timing: a PI loop (`pll.PILoop`) around a detector from `ted` (zero crossing, early-late) | `time.OerderMeyr`, `time.MatTrack` |
| `AGC` | the gain, sample by sample (`mu`, linear or in logs) | `rpps.sync.AGC` (the level per window) |
| `PeakAGC` | the peak, followed and decayed sample by sample | `rpps.sync.PeakAGC` (closed form, the same output) |
| `Squelch` | whether a signal is there, the level followed sample by sample | `rpps.sync.Squelch` (an IIR section, the same output) |
"""
from . import pll, ted
from .pll import PILoop, loop_gains
from .mm import MM
from .tll import TLL
from .agc import AGC, PeakAGC
from .squelch import Squelch
