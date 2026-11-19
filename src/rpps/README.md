# rpps package

The packages, in the order a receiver uses them; README.dev.md has the design, config/README.md the definitions.

| Package | What it holds |
|---|---|
| `source` | sample sources and sinks: files, devices (pysdrlib), memory |
| `signal` | sample-level DSP: filters and pulses (`signal.filters`), resampling, channelizing, channel impairments |
| `sync` | acquisition, AGC, timing, carrier, frame and puncture sync, equalizers |
| `wave` | waveform stages: pulse shaping glue, FM/AM/SSB, OFDM, staggering |
| `symbol` | symbols to channel bits and back: constellations, differential coding, spreading, physical headers |
| `code` | channel bits to data bits: FEC (`code.types`), interleavers, scramblers (`code.scram`), line codes, markers |
| `link` | data bits to packets: CRCs, frames (HDLC), link headers |
| `sequence` | bit sequences shared by the above: recurrences, Gold codes, chip codes, words |
| `profile` | a whole signal from one config file, both directions |
| `analysis` | blind estimation outside the data path: what a signal, code or scrambler is |
| `math` | GF(2) and GF(2^m) arithmetic (polynomials, Berlekamp-Massey, elimination), demapping, probability |
| `base` | `Data`, `Pipe` / `Stage` / `Pipeline`, streams and type tags |
| `parallel` | frame-parallel runners and their kernels |
| `viz` | matplotlib plots (its own style on its own figures) |

Private modules: `_config` (definitions: loading, `extends`, tags), `_native` (the C++ kernels; `_accel.native = None`
runs the Python loops), `_effort` (`rp.set_effort`).
