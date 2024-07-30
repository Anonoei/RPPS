# RPPS
 RF Parameter Processor Suite

RPPS is a generic signal processor/generator library.

 - [Documentation](https://anonoei.github.io/RPPS/)
 - [PyPI](https://pypi.org/project/rpps/)

## Example Usage
```
import rpps as rp

def main():
    mod = rp.mod.name("QPSK", 0) # Use QPSK modulation, with mapping 0
    ecc = rp.coding.name("BLK", "Repetition", 3) # Use Repetition coding, with rate of 2

    pipeline = rp.Pipeline(mod, ecc) # Initialize a processing pipeline

    enc_msg = b"Test" # Define the data to process
    syms = pipeline.enc(enc_msg) # Encode data with ecc, and mod. Get the symbols
    path = pipeline.meta.serialize(syms) # Serialize the symbols to file

    data = pipeline.from_file(path) # Read the symbols and metadata from file, use the same pipeline processing
    print(data) # Check decoded data is what you encoded

if __name__ == "__main__":
  main()
```

## Install
1. Run `python3 -m pip install rpps`
2. In your project, `import rpps as rp`

## Roadmap
 - [ ] [Pre-processing](https://github.com/Anonoei/RPPS/tree/main/src/rpps/freq)
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
     - [ ] LDPC
     - [ ] TPC
   - [ ] Convolutional
     - [ ] Viterbi
 - [ ] De-scramble
   - [ ] v.35
 - [ ] De-frame
   - [ ] HDLC
   - [ ] PPP
