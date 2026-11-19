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
| modulation, rate, offset | `Acquire` (`detect` once); code phase and Doppler: `acquire_code` | |
| channel code, bit map, alignment | `AcquireFEC`: configured codes found on the symbols (`analysis.identify.identify`), turned to the decoder's map, carrier and symbol slips tracked; `est["fec"].rx()` decodes | |

## Second class: per-sample loops
For signals and conditions block estimation does not suit: `time.MM` (Mueller & Muller, Gardner), `time.TLL` (a timing
locked loop around a detector from `time.ted`: `ted="zx"` zero crossing, `"el"` early-late), `Costas`, `AGC`,
`equalize.LMS`, `equalize.CMA`. Profiles reach them by name (`timing: mm`, `timing: {"method": "tll", "ted": "el"}`,
`carrier: costas`, `equalize: lms`). Their loop filter is `pll.PILoop` (gains from a bandwidth: `pll.loop_gains`).

## Known, measured, detected
- `time.Known`: timing that is given, not estimated (a loopback, a source whose timing is known); fuses the matched filter
  before it (`timing: known`).
- `detect`, `Detection`, `rate.find_baud`: the feed-forward estimates `Acquire` runs (rate, timing, modulation, offset,
  phase); blind analysis uses them too.
- `quality`: Es/N0, Eb/N0 and EVM of locked symbols, once (`measure`), over a window (`Track`) or on the stream
  (`Monitor`, `{"sync": "quality"}`).
