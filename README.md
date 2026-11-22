# RPPS
RF Parameter Processor Suite: a generic signal processor/generator library.

- [Documentation](https://anonoei.github.io/RPPS/): guides, examples and the API reference
- [PyPI](https://pypi.org/project/rpps/)
- For AI agents: [AGENTS.md](https://github.com/Anonoei/RPPS/blob/main/AGENTS.md), and
  [llms-full.txt](https://anonoei.github.io/RPPS/llms-full.txt) (the whole library in one file)

## Install
```
python3 -m pip install rpps          # rpps[viz] adds matplotlib for rp.viz
```
Definitions (codes, scramblers, modulations, whole signals) ship with the package. The per-sample loops and decoders run
in C++ with a Python fallback; from a checkout, `python builder.py -c` builds them (pybind11 and a C++17 compiler).

## Usage
Everything is driven by operators: `*` encodes / modulates / scrambles, `/` undoes it, `+` runs a DSP stage. The data
carries its sample rate and the state of every stage from block to block.

```python
import rpps as rp

mod = rp.symbol.load("psk/qpsk")    # or rp.symbol.generate("16QAM"), rp.symbol.PSK(8), rp.symbol.APSK(32)
ecc = rp.code.Hamming(7, 4)         # or rp.code.load("block/hamming/7_4")
scr = rp.code.scram.load("v35")

msg = rp.Data(b"Hello World!")
syms = msg * scr * ecc * mod        # scramble, encode, modulate
out = syms / mod / ecc / scr        # demodulate, decode, descramble
assert out.as_bytes().hex == msg.as_bytes().hex
```

Receiving: stages added together make a pipeline, and blocks flow through it with their state carried along.
```python
# docs: skip
src = rp.source.File("ci16t", "capture.cs16", Fs=100e6, cf=1.0e9)       # or rp.source.Device("hackrf", cf=..., Fs=...)
rx = (rp.signal.filters.ShiftFreq(22.3e6) + rp.signal.filters.low_pass(401, 3e6, 100e6) + rp.signal.sample.down.Decimate(8)
      + rp.sync.Acquire() + rp.sync.time.Mat() + rp.sync.phase.FFCarrier())
for out in src(16384) + rx:
    print(out.Fs, out.sps, out.det)
```

A known signal is a profile: one JSON file describing it in both directions.
```python
# docs: skip
p = rp.profile.load("ble/adv", channel=38)               # python -m rpps ls profile: the shipped ones
for pdu in rp.source.File("cf32", "ch38.cf32", Fs=4e6, cf=2426e6)(65536) + p.rx():
    print(pdu.as_bytes().hex)
wave = p.tx()(rp.Data(bytes(31)), fields={"type": 2})   # the same file transmits
```

From the shell: `python -m rpps ls` (definitions), `python -m rpps steps` (what a profile can name),
`python -m rpps rx ble/adv capture.cf32 --fs 4e6 --param channel=38`.

## Learn more
| | |
|---|---|
| [Guides](https://anonoei.github.io/RPPS/guide/data-model/) | the data model, receiving, profiles, definitions, unknown signals, real time, building on RPPS |
| [Examples](https://anonoei.github.io/RPPS/examples/) | one runnable, self-checking script per recommended way of doing a thing ([examples/](https://github.com/Anonoei/RPPS/tree/main/examples)) |
| [API](https://anonoei.github.io/RPPS/api/) | every package; the design (model, categories, contracts, streaming rules) is `rpps.base` |
| [projects/](https://github.com/Anonoei/RPPS/tree/main/projects) | worked receivers on real signals: FM, BLE, ADS-B, Wi-Fi, ATSC, LTE |
| [Contributing](https://anonoei.github.io/RPPS/contributing/) | setup, tests, native kernels, CI, conventions |
| [Roadmap](https://anonoei.github.io/RPPS/roadmap/) | what is built and what is not yet |
