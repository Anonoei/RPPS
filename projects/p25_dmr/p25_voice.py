"""P25 voice frames (LDU1 / LDU2): the encryption algorithm, key and message indicator, and the call's talkgroup

    python projects/p25_dmr/p25_voice.py data/fed/fed_407.cf64 --fs 4 --cf 407 --freq 406.8375 407.3375 --seconds 20

An LDU is 1728 bits (864 symbols incl. status symbols) after its sync: 9 IMBE voice codewords of 144 bits, 24 ten-bit codewords (Hamming(10,6,3)) = 24 hexbits, and 32 bits of low-speed data.
In an LDU2 the hexbits are the *encryption sync*: message indicator (72 bits), algorithm ID (8 bits; 0x80 = unencrypted), key ID (16 bits), then 8 Reed-Solomon parity hexbits. In an LDU1 they are
the *link control*: opcode, MFID, service options, talkgroup, source unit. Bit positions are those of OP25's op25_imbe_frame.h (imbe_ldu_ls_data_bits). Hexbits are Hamming-corrected, then the
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

LS_BITS = np.array([
    410, 411, 412, 413, 414, 415, 416, 417, 418, 419, 420, 421,
    422, 423, 424, 425, 426, 427, 428, 429, 432, 433, 434, 435,
    436, 437, 438, 439, 440, 441, 442, 443, 444, 445, 446, 447,
    448, 449, 450, 451, 600, 601, 602, 603, 604, 605, 606, 607,
    608, 609, 610, 611, 612, 613, 614, 615, 616, 617, 618, 619,
    620, 621, 622, 623, 624, 625, 626, 627, 628, 629, 630, 631,
    632, 633, 634, 635, 636, 637, 638, 639, 788, 789, 792, 793,
    794, 795, 796, 797, 798, 799, 800, 801, 802, 803, 804, 805,
    806, 807, 808, 809, 810, 811, 812, 813, 814, 815, 816, 817,
    818, 819, 820, 821, 822, 823, 824, 825, 826, 827, 828, 829,
    978, 979, 980, 981, 982, 983, 984, 985, 986, 987, 988, 989,
    990, 991, 992, 993, 994, 995, 996, 997, 998, 999, 1000, 1001,
    1002, 1003, 1004, 1005, 1008, 1009, 1010, 1011, 1012, 1013, 1014, 1015,
    1016, 1017, 1018, 1019, 1168, 1169, 1170, 1171, 1172, 1173, 1174, 1175,
    1176, 1177, 1178, 1179, 1180, 1181, 1182, 1183, 1184, 1185, 1186, 1187,
    1188, 1189, 1190, 1191, 1192, 1193, 1194, 1195, 1196, 1197, 1198, 1199,
    1200, 1201, 1202, 1203, 1204, 1205, 1206, 1207, 1356, 1357, 1358, 1359,
    1360, 1361, 1362, 1363, 1364, 1365, 1368, 1369, 1370, 1371, 1372, 1373,
    1374, 1375, 1376, 1377, 1378, 1379, 1380, 1381, 1382, 1383, 1384, 1385,
    1386, 1387, 1388, 1389, 1390, 1391, 1392, 1393, 1394, 1395, 1396, 1397,
])
_HAMMING = rp.coding.load("block", "hamming/p25_10_6", decoder="syndrome") # Hamming (10,6,3): 6 data bits, 4 parity
_RS = {0x5: rp.coding.load("block", "rs/p25_24_12")._dec, 0xA: rp.coding.load("block", "rs/p25_24_16")._dec} # LDU1 link control, LDU2 encryption sync
ALGO = {0x80: "unencrypted", 0x81: "DES-OFB", 0x84: "AES-256", 0xAA: "ADP (RC4)", 0x9F: "DES-XL", 0x83: "3DES", 0x85: "AES-128"}


def hexbits(bits):
    """24 Hamming(10,6,3)-corrected hexbits of an LDU's link-control / encryption-sync words, from the frame's 1728 bits"""
    words = np.asarray(bits)[LS_BITS].astype(np.uint8)
    data = _HAMMING._dec.decode(words).reshape(24, 6)
    return [int(v) for v in data @ (1 << np.arange(5, -1, -1))]


def rs_correct(hb, duid):
    """Reed-Solomon decode of an LDU's 24 hexbits: (corrected hexbits with the parity left as received, symbols corrected) or (None, -1)"""
    code = _RS[duid]
    info = code.decode(np.array(hb), symbols=True)
    if code.errors[0] < 0:
        return None, -1
    return [int(v) for v in info] + list(hb[code.num:]), int(code.errors[0])


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
