# Real time

```python
# docs: skip
blocks = rp.source.Device("hackrf", cf=915e6, Fs=10e6)(65536).threaded(drop=True)  # the radio is read in a thread of its own
for out in blocks.pipelined(rx):     # and every stage of rx runs in its own thread
    ...
print(out.stream.load())             # time each stage takes, as a fraction of real time (over 1.0: too slow)
```

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
- **Only what you need.** `profile.rx(kinds={...})` builds only the steps the wanted messages need.

The pipelined threads and their load are shown end to end in
[streaming_io.py](receiving.md#sources-sinks-and-carried-state).
