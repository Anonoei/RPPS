"""Look for DMR (ETSI TS 102 361, 4800 baud 4FSK, 12.5 kHz, 2-slot TDMA) bursts in a capture, and read each burst's colour code and data type

    python projects/p25_dmr/dmr_sync.py data/lmr/lmr_462.5.cf64 --fs 8 --cf 462.5 --freq 462.0063 464.4187 --seconds 20

Same front end as p25_sync.py (it imports it): mix the channel to zero, FM demodulate at 10 samples per symbol, re-pick the sampling instant of every burst from its sync, calibrate the levels on the sync's
+-3 symbols. The 48-bit sync is correlated for the base-station voice / data patterns (BS = from a repeater) and the mobile / direct-mode ones; a hit is a burst of a 264-bit (30 ms) slot.
The *slot type* (colour code 4 bits + data type 4 bits, Golay(20,8) protected) is split into two 10-bit halves on either side of the sync; here the 8 information bits are read raw (no Golay correction),
so the statistics over many bursts are what to trust. Data types: 0 PI header, 1 voice LC header, 2 terminator with LC, 3 CSBK, 4 MBC header, 5 MBC continuation, 6 data header, 7 rate 1/2 data, 8 rate 3/4 data,
9 idle, 10 rate 1 data. Colour code 0-15 separates repeaters on one frequency. No payload is decoded (no BPTC, no voice).
Meant to be upstreamed into RPPS.
"""
import argparse
import os
import sys
from collections import Counter

import numpy as np

import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); import _paths  # noqa: E402,F401
import p25_sync

_LEV = {1: 3.0, 0: 1.0, 2: -1.0, 3: -3.0}


def pattern(hexs):
    b = format(int(hexs, 16), "048b")
    return np.array([_LEV[int(b[i:i + 2], 2)] for i in range(0, 48, 2)])


SYNCS = {"BS voice": pattern("755FD7DF75F7"), "BS data": pattern("DFF57D75DF5D"), "MS voice": pattern("7F7D5DD57DFD"), "MS data": pattern("D5D7F77FD757"),
         "direct TS1 voice": pattern("5D577F7757FF"), "direct TS1 data": pattern("F7FDD5DDFD55"), "direct TS2 voice": pattern("7DFFD5F55D5F"), "direct TS2 data": pattern("D7557F5FF7F5")}
TYPES = {0: "PI header", 1: "voice LC header", 2: "terminator LC", 3: "CSBK", 4: "MBC header", 5: "MBC cont.", 6: "data header", 7: "rate 1/2 data", 8: "rate 3/4 data", 9: "idle", 10: "rate 1 data"}


def slot_type(fr):
    """(colour code, data type) from 34 level-calibrated symbols: 5 before the sync, the 24 of the sync, 5 after"""
    d = p25_sync.slice_dibits(np.concatenate([fr[:5], fr[29:34]]))
    bits = "".join(format(int(v), "02b") for v in d)
    return int(bits[:4], 2), int(bits[4:8], 2)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file"); ap.add_argument("--fs", type=float, required=True, help="[MS/s]"); ap.add_argument("--cf", type=float, required=True, help="[MHz]")
    ap.add_argument("--freq", type=float, nargs="+", required=True); ap.add_argument("--seconds", type=float, default=10.0); ap.add_argument("--corr", type=float, default=0.9); ap.add_argument("--timeline", action="store_true", help="list the non-CSBK bursts in time order (call headers, terminators, PI headers, voice)")
    a = ap.parse_args(argv)
    fs = a.fs*1e6; x = np.fromfile(a.file, np.complex64, count=int(fs*a.seconds))
    for f in a.freq:
        sn, c = p25_sync.symbols(x, fs, a.cf*1e6, f*1e6); stream = p25_sync.symbols.stream
        found = []
        for name, P in SYNCS.items():
            hits, r = p25_sync.sync_hits(sn, a.corr, P)
            found += [(i, name, r[i]) for i in hits]
        if not found:
            print(f"{f:9.4f} MHz: no DMR sync"); continue
        rows = []; found.sort(); kinds = Counter(n for _, n, _ in found); cc = Counter(); dt = Counter(); gaps = []
        prev = None
        for i, name, rr in found:
            fr = p25_sync.refine(stream, i, n=29, sync=SYNCS[name], before=5)
            st = None
            if len(fr) == 34:
                st = slot_type(fr/(np.mean(fr[5:29]*SYNCS[name])/9.0))
            if st:
                cc[st[0]] += 1; dt[st[1]] += 1
                if a.timeline and st[1] != 3:
                    rows.append(f"             {i/4800:7.3f} s  {name:9s} cc {st[0]:2d}  {TYPES.get(st[1], st[1])}")
            if prev is not None:
                gaps.append((i - prev)/4800*1e3)
            prev = i
        print(f"{f:9.4f} MHz: {len(found)} bursts in {a.seconds:g} s; sync kinds {dict(kinds)}; median spacing {np.median(gaps):.0f} ms")
        for r_ in rows:
            print(r_)
        print(f"             colour code (raw) {cc.most_common(4)}; data type (raw) {[(TYPES.get(k, k), v) for k, v in dt.most_common(5)]}")


if __name__ == "__main__":
    main()
