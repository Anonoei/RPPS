# sync

Estimates something unknown about the data and corrects or aligns it, at whatever representation it can be separated
(README.dev.md, Sync methods). Its estimates go to `stream.est[name]`, never into the data.

## First class: block estimation, tracked across windows
Windows at absolute stream positions, so any block cut gives the same output; a slow loop (`track`) carries each parameter
and its rate from window to window, gates outliers and coasts through gaps; the correction is evaluated per item.

| Estimates | Operator | Profile method |
|---|---|---|
| symbol timing | `time.OerderMeyr`: square-law, cubic interpolation | `timing: oerder_meyr` (default) |
| carrier frequency and phase | `phase.MPower`: M-th power of the phase, FFT acquisition | `carrier: mpower` (default) |
| channel | `equalize.BlockLS`: least squares per window against decisions | `equalize: ls` (default) |
| carrier phase per frame | `PilotPhase` (pilot blocks), `phase.FFCarrier` (Viterbi & Viterbi) | `carrier: pilots`, `carrier: ff` |
| frame starts | `FrameSync` (a preamble and the best word of a header code), `Delimiter` (a word among bits) | `frame`, `delimit` |
| alignment on bits | `Align` (markers, patterns, polarity), `Puncture` (puncture phase) | `align`, `puncture` |
| modulation, rate, offset | `Acquire`; code phase and Doppler: `acquire_code` | |

## Second class: per-sample loops
For signals and conditions block estimation does not suit: `time.MM` (Mueller & Muller, Gardner), `Costas`, `PLL`, `AGC`,
`equalize.LMS`, `equalize.CMA`. Profiles reach them by name (`timing: mm`, `carrier: costas`, `equalize: lms`). Their
error detectors are in `time/ed`, `phase/ed`, `freq/ed` and `ed.py`: data aided, decision directed or non-data aided.
