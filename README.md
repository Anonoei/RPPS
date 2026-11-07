# RPPS
 RF Parameter Processor Suite

RPPS is a generic signal processor/generator library.
 - [Documentation](https://anonoei.github.io/RPPS/)
 - [PyPI](https://pypi.org/project/rpps/)

## Usage

Everything is driven by operators: `*` encodes / modulates / scrambles, `/` undoes it, `+` runs a DSP stage.
The data carries its sample rate and the state of every stage, so there is nothing to set up or reset.

### Bits to symbols and back ([file](https://github.com/Anonoei/RPPS/tree/main/tests/example.py))
```python
import rpps as rp

mod = rp.mod.load("QPSK")           # first bit map; load("QPSK", "R090 inverted") for a rotated / spectrally inverted one. Or rp.mod.QAM(16), rp.mod.PSK(8), rp.mod.APSK(32), rp.mod.FSK(4)...
ecc = rp.coding.Hamming(7, 4)       # or rp.coding.load("block", "hamming.7_4")
scr = rp.scram.load("fdt", "v35")

msg = rp.Data(b"Hello World!")
syms = msg * scr * ecc * mod        # Scramble, encode, modulate
out = syms / mod / ecc / scr        # Demodulate, decode, descramble
assert out.as_bytes().hex == msg.as_bytes().hex

# Pipes multiplied together make a reusable link, decoded in the opposite order
link = scr * ecc * mod
syms = msg * link
out = syms / link
```

### Receiving a signal
```python
# docs: skip
src = rp.serial.File("ci32t", "capture.cs16", Fs=100e6, cf=1.0e9)       # or rp.serial.Device("hackrf", cf=..., Fs=...)
rx = (rp.filters.ShiftFreq(22.3e6)  + rp.filters.low_pass(401, 3e6, 100e6) + rp.sample.down.Decimate(8)
      + rp.sync.Acquire() + rp.sync.time.Mat() + rp.sync.phase.FFCarrier())     # stages added together make a pipeline

for out in src(16384) + rx:          # blocks flow through, state carried from one to the next
    print(out.Fs, out.sps, out.det)  # rates follow the stages; Acquire's detection is on the data

src(16384) + rx >> rp.serial.Sink("cf64", "symbols.iq")       # or straight into a file
```
A stage can also be applied by hand to a block: `blk + rp.filters.ShiftFreq(1e3)`.
`rp.channel` has impairments to test against (`AWGN`, `CFO`, `Fading`, `Multipath`, `ClockDrift`...).

### Real-time spectrum
![RT](/media/example_rt.jpg?raw=true "Real-time Persistent Spectrum")
- [file](https://github.com/Anonoei/RPPS/tree/main/tests/dev/viz_rt.py)

## Install
1. Run `python3 -m pip install rpps` (`rpps[viz]` adds matplotlib for `rp.viz`)
2. In your project, `import rpps as rp`

Definitions (codes, scramblers, modulations) ship with the package. Add your own without touching it:
`rp.add_config_path("my_config")` (laid out like `src/rpps/config`) or set `RPPS_CONFIG`.
Logging is off until you ask: `rp.logger.enable("DEBUG")`.

## Roadmap
 - [ ] [Serial](https://github.com/Anonoei/RPPS/tree/main/src/rpps/serial)
   - [X] File
   - [ ] (Linux only) tun/tap
   - [ ] Socket
 - [ ] [Filters](https://github.com/Anonoei/RPPS/tree/main/src/rpps/filters)
   - [X] Pulse Shaping
   - [X] Window functions
 - [X] [Sample](https://github.com/Anonoei/RPPS/tree/main/src/rpps/sample)
 - [ ] [Sync](https://github.com/Anonoei/RPPS/tree/main/src/rpps/sync)
   - [X] Frequency
   - [ ] Phase
   - [ ] Time
 - [X] [Mod](https://github.com/Anonoei/RPPS/tree/main/src/rpps/mod)
   - [X] PSK
   - [X] QAM
   - [X] APSK
   - [X] ASK
   - [X] FSK
 - [ ] [Coding](https://github.com/Anonoei/RPPS/tree/main/src/rpps/coding)
   - [ ] Block
     - [X] Repetition
     - [X] Hamming (decoded on its syndrome trellis)
     - [X] BCH
     - [X] Reed-Solomon
     - [X] LDPC (min-sum, DVB-S2 tables)
     - [ ] TPC
   - [X] Convolutional
     - [X] Viterbi (hard and soft)
     - [X] BCJR / trellis (log-MAP, max-log-MAP)
   - [X] Soft decisions (demodulator LLRs into the decoders)
 - [ ] [Scram](https://github.com/Anonoei/RPPS/tree/main/src/rpps/scram)
   - [X] fibonacci
   - [X] galois
   - [X] V.35
 - [ ] [Encapsulation](https://github.com/Anonoei/RPPS/tree/main/src/rpps/encap)
   - [X] CRC
   - [X] HDLC
   - [ ] PPP
 - [ ] [Viz](https://github.com/Anonoei/RPPS/tree/main/src/rpps/viz)
   - [X] Real-Time (STFT)
   - [X] Over-Time

## Contributing
 1. `git clone https://github.com/Anonoei/RPPS`
 2. `cd RPPS`
 3. `git branch -c feature/<your feature>`
 4. `python3 builder.py -b -l` build and install rpps locally
    - This also installs deps, and dev_deps automatically
 5. run `python3 -m pytest` (add `-m "not slow"` to skip the chain tests). Scratch scripts live in `tests/dev`
 6. Check out the `dev` branch for latest changes
