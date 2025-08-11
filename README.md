# RPPS
 RF Parameter Processor Suite

RPPS is a generic signal processor/generator library.
 - [Documentation](https://anonoei.github.io/RPPS/)
 - [PyPI](https://pypi.org/project/rpps/)

## Examples & Demos

### Example Usage
- [file](https://github.com/Anonoei/RPPS/tree/main/tests/example.py)
```
import rpps as rp

mod = rp.mod.load("QPSK")
mod.set_mapping(mod.get_maps()[0])
ecc = rp.coding.load("blk", "hamming.7_4")
scr = rp.scram.load("fdt", "v35")

enc_msg = rp.Data(b"Hello World!")

f_pipe = lambda inp:inp * scr * ecc * mod
r_pipe = lambda syms:syms / mod / ecc / scr

syms = f_pipe(enc_msg) # Encode data with ecc, and mod. Get the symbols

data = r_pipe(syms) # Read the symbols

dec_msg = rp.Data(data)
print(f"{enc_msg.hex == dec_msg.hex}") # Check decoded data is what you encoded

```

### Real-time spectrum
![RT](/media/example_rt.jpg?raw=true "Real-time Persistent Spectrum")
- [file](https://github.com/Anonoei/RPPS/tree/main/tests/example_rt.py)
```
import rpps as rp

sink = rp.serial.File(<format>, <path>)
meta = rp.Meta(Fs=<>, cf=<>)
sweep_time, display = 10, 50 # ms
num_samps = int((sweep_time/1000)*meta.Fs)
nfft, overlap, vbw = 1024, 0.8, 1000
window = "blackman"
fig, ax = plt.subplots()
im = ax
for samps in sink(num_samps):
    snips = rp.utils.stft.stft(samps, nfft, overlap, window)
    psds = rp.utils.stft.psd(snips, meta.Fs, vbw)
    im = rp.viz.rt.Persistent(im, psds)
    plt.pause(display/1000)
```

## Install
1. Run `python3 -m pip install rpps`
2. In your project, `import rpps as rp`

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
   - [ ] Frequency
   - [ ] Phase
   - [ ] Time
 - [ ] [Mod](https://github.com/Anonoei/RPPS/tree/main/src/rpps/mod)
   - [X] PSK
   - [ ] QAM
   - [ ] APSK
   - [ ] ASK
   - [X] FSK
 - [ ] [Coding](https://github.com/Anonoei/RPPS/tree/main/src/rpps/coding)
   - [ ] Block
     - [X] Repetition
     - [X] Hamming
     - [ ] TPC
     - [ ] LDPC
   - [ ] Convolutional
     - [ ] Viterbi
 - [ ] [Scram](https://github.com/Anonoei/RPPS/tree/main/src/rpps/scram)
   - [X] fibonacci
   - [X] galois
   - [X] V.35
 - [ ] Frame
   - [ ] HDLC
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
 5. run `python3 tests/dev.py`
 6. Check out the `dev` branch for latest changes
