# RPPS
 RF Parameter Processor Suite

RPPS is a generic signal processor/generator library.

 - [Documentation](https://anonoei.github.io/RPPS/)
 - [PyPI](https://pypi.org/project/rpps/)

## Example Usage
```
import rpps as rp

def main():
    mod = rp.mod.load("QPSK")
    mod.set_mapping(mod.get_maps()[0])
    ecc = rp.coding.load("blk", "hamming.7_4")
    scr = rp.scram.load("fdt", "v35")

    enc_msg = rp.dobject.StreamData(b"Hello World!")

    f_pipe = lambda inp:inp * scr * ecc * mod
    r_pipe = lambda syms:syms / mod / ecc / scr

    syms = f_pipe(enc_msg) # Encode data with ecc, and mod. Get the symbols

    data = r_pipe(syms) # Read the symbols

    dec_msg = rp.dobject.StreamData(data)
    print(f"{enc_msg.hex == dec_msg.hex}") # Check decoded data is what you encoded

if __name__ == "__main__":
  main()
```

## Install
1. Run `python3 -m pip install rpps`
2. In your project, `import rpps as rp`

## Roadmap
 - [ ] [Interfaces]()
   - [ ] File
   - [ ] (Linux only) tun/tap
   - [ ] Socket
 - [ ] [Pre-processing]()
   - [ ] Filters
   - [ ] Pulse Shaping
 - [ ] [Modulation](https://github.com/Anonoei/RPPS/tree/main/src/rpps/mod)
   - [ ] PSK
     - [X] BPSK
     - [X] QPSK
     - [X] 8PSK
   - [ ] QAM
   - [ ] APSK
   - [ ] ASK
   - [ ] FSK
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

## Contributing
 1. `git clone https://github.com/Anonoei/RPPS`
 2. `cd RPPS`
 3. `git branch -c feature/<your feature>`
 4. `python3 builder.py -b -l` build and install rpps locally
    - This also installs deps, and dev_deps automatically
 5. run `python3 tests/dev.py`
 6. Check out the `dev` branch for latest changes
