"""Second class: per-sample loops (rpps.sync)

Each updates its estimate once a sample or a symbol, carrying it from one to the next. Block estimation tracked across
windows (rpps.sync's first class) replaces them in the profiles and examples; they stay for signals and conditions it
does not suit, as references, and for comparison. Each runs in the C++ kernels when they are built (rpps._native), with
its Python loop as the reference.

| Loop | Estimates | First-class counterpart |
|---|---|---|
| `Squelch` | whether a signal is there, the level followed sample by sample | `rpps.sync.Squelch` (an IIR section, the same output) |
"""
from .squelch import Squelch
