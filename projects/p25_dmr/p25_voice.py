"""P25 voice frames (LDU1 / LDU2): the encryption algorithm, key and message indicator, and the call's talkgroup

    python projects/p25_dmr/p25_voice.py data/fed/fed_407.cf64 --fs 4 --cf 407 --freq 406.8375 407.3375 --seconds 20

An LDU is 1728 bits (864 symbols incl. status symbols) after its sync: 9 IMBE voice codewords of 144 bits, 24 ten-bit codewords (Hamming(10,6,3)) = 24 hexbits, and 32 bits of low-speed data.
In an LDU2 the hexbits are the *encryption sync*: message indicator (72 bits), algorithm ID (8 bits; 0x80 = unencrypted), key ID (16 bits), then 8 Reed-Solomon parity hexbits. In an LDU1 they are
the *link control*: opcode, MFID, service options, talkgroup, source unit. Bit positions follow from the LDU layout of TIA-102.BAAA (`_ls_bits`). Hexbits are Hamming-corrected, then the
Reed-Solomon code is applied (RS(24,12,13) on an LDU1, RS(24,16,9) on an LDU2, rpps definitions): only frames that decode are counted. Meant to be upstreamed into RPPS.
"""
import argparse
import os
import sys
from collections import Counter

import numpy as np

import rpps as rp

import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); import _paths  # noqa: E402,F401
import p25_sync

def _ls_bits():
    """Positions in the LDU (from the frame sync on, status symbols included) of its 240 link control / encryption sync bits,
    from the LDU layout of TIA-102.BAAA: sync 48, NID 64, voice codewords 1 and 2 (144 bits each), then six times 40 bits
    (four Hamming (10,6,3) words) and a voice codeword (3..8), then 32 bits of low-speed data and voice codeword 9 = 1680
    bits. A status symbol (2 bits) follows every 70 bits, which makes the 1728 bits sent"""
    pos, at = [], 48 + 64 + 2*144
    for _ in range(6):
        pos += range(at, at + 40)
        at += 40 + 144
    pos = np.array(pos)
    return pos + 2*(pos//70)


LS_BITS = _ls_bits()
_HAMMING = rp.code.load("block/hamming/p25_10_6", decoder="syndrome") # Hamming (10,6,3): 6 data bits, 4 parity
_RS = {0x5: rp.code.load("block/rs/p25_24_12").decoder, 0xA: rp.code.load("block/rs/p25_24_16").decoder} # LDU1 link control, LDU2 encryption sync
ALGO = {0x80: "unencrypted", 0x81: "DES-OFB", 0x84: "AES-256", 0xAA: "ADP (RC4)", 0x9F: "DES-XL", 0x83: "3DES", 0x85: "AES-128"}


def hexbits(bits):
    """24 Hamming(10,6,3)-corrected hexbits of an LDU's link-control / encryption-sync words, from the frame's 1728 bits"""
    words = np.asarray(bits)[LS_BITS].astype(np.uint8)
    data = _HAMMING.decoder.decode(words).reshape(24, 6)
    return [int(v) for v in data @ (1 << np.arange(5, -1, -1))]


def rs_correct(hb, duid):
    """Reed-Solomon decode of an LDU's 24 hexbits: (corrected hexbits with the parity left as received, symbols corrected) or (None, -1)"""
    code = _RS[duid]
    info, errors = code.decode(np.array(hb), symbols=True, status=True)
    if errors[0] < 0:
        return None, -1
    return [int(v) for v in info] + list(hb[code.num:]), int(errors[0])


def ess(hb):
    """(message indicator hex, algorithm ID, key ID) from an LDU2's hexbits"""
    mi = 0
    for h in hb[:12]:
        mi = (mi << 6) | h
    return f"{mi:018x}", (hb[12] << 2) | (hb[13] >> 4), ((hb[13] & 0xF) << 12) | (hb[14] << 6) | hb[15]


def link_control(hb):
    """(LCF, MFID, service options, talkgroup, source) from an LDU1's hexbits (LCF 0 = group voice channel user)"""
    v = 0
    for h in hb[:12]:
        v = (v << 6) | h
    return (v >> 64) & 0xFF, (v >> 56) & 0xFF, (v >> 48) & 0xFF, (v >> 24) & 0xFFFF, v & 0xFFFFFF


def frame_bits_full(fr):
    """1728 bits of an LDU: all 864 symbols of the level-calibrated frame, status symbols kept"""
    d = p25_sync.slice_dibits(fr[:864])
    return np.array([int(b) for v in d for b in format(int(v), "02b")], np.uint8)


def ldus(sn, stream):
    """Yield (duid, calibrated 864-symbol frame) for each LDU1 / LDU2 found in the symbol stream"""
    hits, _ = p25_sync.sync_hits(sn)
    for i in hits:
        fr = p25_sync.refine(stream, i, n=864)
        if len(fr) < 864:
            continue
        v = p25_sync.nid(fr, 0)
        if v and v[1] in (0x5, 0xA):
            yield v[1], fr/(np.mean(fr[:24]*p25_sync.SYNC)/9.0)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file"); ap.add_argument("--fs", type=float, required=True, help="[MS/s]"); ap.add_argument("--cf", type=float, required=True, help="[MHz]")
    ap.add_argument("--freq", type=float, nargs="+", required=True); ap.add_argument("--seconds", type=float, default=20.0)
    a = ap.parse_args(argv)
    fs = a.fs*1e6; x = np.fromfile(a.file, np.complex64, count=int(fs*a.seconds))
    for f in a.freq:
        sn, c = p25_sync.symbols(x, fs, a.cf*1e6, f*1e6)
        algs, kids, tgs, n1, n2, rs_ok, fixed = Counter(), Counter(), Counter(), 0, 0, 0, 0
        for duid, fr in ldus(sn, p25_sync.symbols.stream):
            hb, ne = rs_correct(hexbits(frame_bits_full(fr)), duid)
            if hb is None:
                n1 += duid == 0x5; n2 += duid == 0xA
                continue
            rs_ok += 1; fixed += ne
            if duid == 0xA:
                n2 += 1; _, alg, kid = ess(hb); algs[alg] += 1; kids[kid] += 1
            else:
                n1 += 1; lcf, mfid, svc, tg, src = link_control(hb)
                if lcf == 0:
                    tgs[(tg, src, mfid, svc)] += 1
        print(f"{f:9.4f} MHz: {n1} LDU1, {n2} LDU2; Reed-Solomon ok {rs_ok}/{n1 + n2} ({fixed} symbols corrected)")
        if algs:
            print("             algorithm ID: " + ", ".join(f"{k:#04x} {ALGO.get(k, '?')} x{n}" for k, n in algs.most_common(4)) + "; key ID: " + ", ".join(f"{k:#06x} x{n}" for k, n in kids.most_common(3)))
        for (tg, src, mfid, svc), n in tgs.most_common(3):
            print(f"             link control: talkgroup {tg}, source unit {src}, MFID {mfid:#04x}, service options {svc:#04x}  x{n}")


if __name__ == "__main__":
    main()
