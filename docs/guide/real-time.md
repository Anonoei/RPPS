# Real time

```python
# docs: skip
blocks = rp.source.Device("hackrf", cf=915e6, Fs=10e6)(65536).threaded(drop=True)  # the radio is read in a thread of its own
for out in blocks.pipelined(rx):     # and every stage of rx runs in its own thread
    ...
print(out.stream.load())             # each operator's time as a fraction of real time (over 1.0: too slow)
print(out.stream.realtime(window=2)) # one number, over the last 2 s of signal: the busiest thread (<= 1.0 keeps up)
```

- **Load.** `stream.load()` is each operator's own time over the signal's: stages (`+`) and pipes (`/`, `*`, the
  decoders) alike, and a parallel region's workers. 1.0 is exactly real time. `load(window=s)` measures over the last
  `s` seconds of signal, so a stage that slows down shows at once. `stream.realtime()` adds the loads up per thread and
  gives the busiest: serial work is the sum, `pipelined` work the slowest stage, a parallel region its workers' time
  over their number.

- **Native kernels.** The per-sample loops (AGC, Costas, timing recovery, equalizers, IIR, demodulators) and the decoders
  (Viterbi, trellis, BCJR / turbo, LDPC, Reed-Solomon, BCH, polar) run in C++ (`rpps._native`, built with the package)
  and release the GIL, so threads run them in parallel; batches of codewords are spread over the cores (`RPPS_THREADS`
  sets how many). Without a compiler RPPS uses its Python loops, the reference (`RPPS_NO_NATIVE=1` forces them).
  Compare with `python tools/bench_stages.py`.
- **Frame-parallel profiles.** After frame sync and the header, a profile's per-frame work can run on threads or
  processes in batches: [Profiles](profiles.md#more-than-one-core), [`rpps.parallel`](../api/parallel.md).
- **Effort.** `rp.set_effort("fast" | "balanced" | "max")`, `with rp.effort(...)`, `RPPS_EFFORT` or
  `rpps rx --effort` trade speed for recovered SNR: decoder iterations, exact or max-log demapping, the timing
  interpolator. The levels and their measured gains are in [`rpps`](../api/rpps.md#rpps._effort).
- **Gaps.** Samples the radio lost (an overrun) and blocks `threaded(drop=True)` discarded are counted in `meta.pos`, at
  the source and through every stage, so positions and `meta.time_of(k)` stay true to time across a gap;
  `out.stream.dropped` is how many source samples were lost. `Device` dates the stream from the radio's clock where it
  timestamps its samples (else the host's).
- **GPUs.** A frame-parallel region's kernels run on any array module: `xp, _ = rpps._accel.device()` gives CuPy
  (built for CUDA or ROCm/HIP; nothing in RPPS names a vendor) or numpy without a GPU, and
  `Parallel("thread", xp=xp)` runs the region on it, one upload and one download a batch. How the hand-off works:
  [`rpps.parallel`](../api/parallel.md).
- **Only what you need.** `profile.rx(kinds={...})` builds only the steps the wanted messages need.

The pipelined threads and their load are shown end to end in
[streaming_io.py](receiving.md#sources-sinks-and-carried-state).

## More than one signal

When one process is not enough (`stream.realtime()` over 1.0), there are three ways to split the work:

1. **A process per signal, each reading the radio.** On a pysdrlib server's machine every RPPS process opens
   `Device("remote")`. The server shares its samples in memory, so each process reads at its own pace, and each does
   its own wideband downconversion.
2. **One channelizer, a process per channel.** The wideband work is done once: one process publishes each channel to
   shared memory (`rp.source.ShmSink`, a pysdrlib ring per channel, a manifest beside it), and each demodulator reads
   its channel (`rp.source.Shm(name)`). Positions stay true to time, a retune starts the readers over, and a live view
   can read the same channels.
3. **One process, a thread per channel**, for a few channels: `Channelizer`, then each channel's stages.

```python
--8<-- "examples/channel_shm.py"
```
