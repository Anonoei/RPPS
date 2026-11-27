"""The RDS receiver of e2e.py as a profile, built in code and run: the same steps, named instead of wired by hand

A profile is a signal's steps in receive order (here a dict; shipped ones are JSON files in config/profile, loaded with
rp.profile.load). `p.rx()` runs them block by block, carrying every stage's state; the runner gives LLRs (positive: 0).
RDS: a 57 kHz BPSK subcarrier of the FM multiplex at 1187.5 bit/s, biphase (Manchester) and differentially coded, in
26-bit blocks (16 data bits and a 10-bit checkword with an offset word: A, B, C, C', D). The blocks are found here by
their checkwords, and the station's name (Program Service, type 0 groups) put together from them.

    python examples/e2e_profile.py [capture.iq]     # the pysdr.org RDS capture: cf32, 250 kS/s, 99.5 MHz
"""
import sys

import numpy as np

import rpps as rp

# From https://pysdr.org/content/rds.html
RDS = {
    "comment": "RDS on an FM broadcast: the 57 kHz subcarrier, biphase BPSK at 1187.5 bit/s, differentially coded",
    "ref": "IEC 62106 (RDS); pysdr.org/content/rds.html",
    "steps": [
        {"wave": "fm", "deviation": 75e3},                     # the FM multiplex
        {"signal": "shift", "freq": -57e3},                    # the RDS subcarrier to 0 Hz
        {"signal": "lowpass", "cutoff": 7.5e3, "taps": 101},
        {"signal": "decimate", "ratio": 10},                   # 25 kS/s
        {"signal": "resample", "up": 19, "down": 25},          # 19 kS/s: 16 samples a bit
        {"signal": "class", "class": "rpps.signal.filters.ShapingBiphase"}, # matched to the biphase symbol
        {"sync": "timing"},                                    # block estimation (Oerder & Meyr), tracked across windows
        {"sync": "carrier", "order": 2},                       # the square's line (M-th power, M = 2), tracked likewise
        {"sync": "quality", "constellation": "psk/bpsk"},      # Es/N0 and EVM, in stream.est["quality"]
        {"symbol": "differential"},                            # a change of phase is a 1
        {"symbol": "constellation", "name": "psk/bpsk"},
    ],
}

# The checkword: the remainder by g(x) = x^10 + x^8 + x^7 + x^5 + x^4 + x^3 + 1, plus the block's offset word, so the
# remainder of a whole received block is its offset word
G = 0b10110111001
OFFSETS = {0x0FC: "A", 0x198: "B", 0x168: "C", 0x350: "C'", 0x1B4: "D"}

def remainder(bits):
    r = 0
    for b in bits:
        r = (r << 1) | int(b)
        if r >> 10:
            r ^= G
    return r

def blocks(bits):
    """(position, offset name, 16 data bits as an int) of every block whose checkword holds"""
    out = []
    for i in range(len(bits) - 25):
        name = OFFSETS.get(remainder(bits[i:i + 26]))
        if name:
            out.append((i, name, int("".join(map(str, bits[i:i + 16])), 2)))
    return out

def in_sync(found):
    """The blocks with another one exactly a block before or after: the stream of blocks, not chance matches"""
    at = {i for i, _, _ in found}
    return [f for f in found if f[0] - 26 in at or f[0] + 26 in at]

def program_service(found):
    """The station name: type 0 groups carry two of its eight characters in block D, at the segment block B names"""
    name = ["?"]*8
    at = {i: (n, v) for i, n, v in found}
    for i, n, b in found:
        if n != "B" or b >> 12 != 0 or (i + 52) not in at or at[i + 52][0] != "D": # B, then D two blocks on
            continue
        seg, d = b & 3, at[i + 52][1]
        name[2*seg:2*seg + 2] = chr(d >> 8), chr(d & 0xFF)
    return "".join(name)

def main(path="data/fm_rds_250k_1Msamples.iq"):
    p = rp.profile.Profile("rds/example", RDS) # (rp.profile.load("family/mode") for a shipped one)
    src = rp.source.File("cf32", path, Fs=250e3, sps=250e3/1187.5, cf=99.5e6)

    bits, out = [], None
    for out in src(16384) + p.rx(): # any block size: the runner carries each stage's state
        bits.append(out.as_hard())
    bits = np.concatenate(bits).astype(np.uint8)
    found = in_sync(blocks(bits))
    print(f"{len(bits)} bits, {out.stream.est['quality']}")
    print(f"{len(found)} RDS blocks in sync ({len(found)*26/len(bits)*100:.0f}% of the bits), station {program_service(found)!r}")
    assert len(found)*26 > 0.9*len(bits), "the RDS blocks should cover the capture"

if __name__ == "__main__":
    main(*sys.argv[1:])
