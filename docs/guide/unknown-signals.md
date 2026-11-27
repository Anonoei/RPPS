# Unknown signals

`rp.analysis` stands outside the data path: it says what a capture holds and what each signal is, so a receiver (often a
profile) can be built for it. `survey(source)` finds the signals in a capture: where, how wide, how strong, roughly what
symbol rate. `profile(source)` takes each one through the receive chain step by step and says which steps process it,
which profile decodes it, and what is still unsolved.

```python
--8<-- "examples/unknown_signal.py"
```

From the shell: `python -m rpps survey capture.cf32 --fs ...`, `python -m rpps profile ...`, `python -m rpps solve ...`
(`-h` on each). On a live radio: [examples/live_profile.py](https://github.com/Anonoei/RPPS/blob/main/examples/live_profile.py).

## Boxes on the signals

`find_signals` (which `survey` runs) is the box on each signal alone: edges, SNR over what the signal stands on, and
when it is on. Its false boxes are set by a false-alarm rate (`pfa`), not a threshold in dB. `Finder` does the same
block by block for a display, keeping each signal's id; with about a second of frames held, an update costs a few
milliseconds and its boxes hardly change from one to the next. A signal it no longer finds stays, `stale`, for its
`hold`.

```python
--8<-- "examples/finding_signals.py"
```

How it works (the floor, the detector, the levels it reads signals at) is in
[`rpps.analysis.spectrum.find`](../api/analysis.md).

## The blind tools

Each blind tool states its statistic and chance level, so "nothing found" is an answer. Which tool answers which question
(symbol rate, constellation, FSK index, OFDM pilots, convolutional and block codes, scramblers, interleavers, frame
markers) is tabled in [`rpps.analysis`](../api/analysis.md). What they find can be saved as a definition
(`Guess.save`), and `rp.sync.AcquireFEC` finds a configured code on a live stream.
