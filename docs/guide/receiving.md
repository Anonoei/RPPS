# Receiving a signal

Stages added together make a pipeline; blocks from a source flow through it with every stage's state carried from one
block to the next, so the output does not depend on how the stream is cut.

```python
# docs: skip
import rpps as rp

src = rp.source.File("ci16t", "capture.cs16", Fs=100e6, cf=1.0e9)       # or rp.source.Device("hackrf", cf=..., Fs=...)
rx = (rp.signal.filters.ShiftFreq(22.3e6) + rp.signal.filters.low_pass(401, 3e6, 100e6) + rp.signal.sample.down.Decimate(8)
      + rp.sync.Acquire() + rp.sync.time.Mat() + rp.sync.phase.FFCarrier())

for out in src(16384) + rx:          # blocks flow through, state carried from one to the next
    print(out.Fs, out.sps, out.det)  # rates follow the stages; Acquire's detection is on the data

src(16384) + rx >> rp.source.Sink("cf32", "symbols.iq")       # or straight into a file
```

A stage can also be applied by hand to one block: `blk + rp.signal.filters.ShiftFreq(1e3)`.

| Package | Gives |
|---|---|
| [`rpps.source`](../api/source.md) | `File`, `SigMF`, `Wav`, `Socket`, `Memory`, `Device` (a radio through pysdrlib); `Sink`; `formats` by pysdrlib's names (`cf32`, `ci16t`, ...) |
| [`rpps.signal`](../api/signal.md) | filters (designed in Hz, or definitions), resampling, `channelize`, DC block, de-emphasis; `signal.channel`: AWGN, CFO, PhaseOffset, ClockDrift, Multipath, Fading, Gain |
| [`rpps.sync`](../api/sync.md) | acquisition, AGC, timing, carrier, equalizers, frame sync; block estimators first, per-sample loops second |
| [`rpps.wave`](../api/wave.md) | matched filters, FM/AM/SSB demodulation, OFDM, staggering |

## A simulated channel, and recovery with block estimators

Sync is first class when it estimates from a window with block operations (FFTs, correlation, least squares) and a
slow loop carries the estimate from window to window; [`rpps.sync`](../api/sync.md) has the rules and the list.

```python
--8<-- "examples/channel_and_sync.py"
```

## Sources, sinks and carried state

Iterating a source carries `stream` and `ctx` for you; calling stages by hand, pass them on yourself.

```python
--8<-- "examples/streaming_io.py"
```
