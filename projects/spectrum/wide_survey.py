"""Wideband RF survey with a HackRF: sweep, confirm, group by allocation, compare runs

    python projects/spectrum/wide_survey.py sweep runs/a            # 30 MHz - 6 GHz, two passes (tile grid shifted 8 MHz): runs/a_1.npz, runs/a_2.npz (~3.5 min)
    python projects/spectrum/wide_survey.py sweep runs/b --lo 700 --hi 900
    python projects/spectrum/wide_survey.py sweep runs/c --lo 1390 --hi 1530 --gain 0.4   # fixed lower gain: where the HackRF's own tune-following plateaus show up
    python projects/spectrum/wide_survey.py report runs/a                  # confirmed emitters grouped by what is allocated there
    python projects/spectrum/wide_survey.py report runs/b --compare runs/a  # what is new, what is gone, what is in both

A sweep is 20 MS/s tiles of 16 MHz; per tile the gain is stepped down the ladder until nothing clips (a detection setting: not the per-band gain search of gain_search.py), the DC spike
of the tile is masked, the power spectrum is stitched. Why two passes: the HackRF has spurs tied to its tiling (multiples of 20 MHz) and edge artifacts of each tile. A real signal sits at the same absolute
frequency whatever the tile grid, so a detection counts only where both passes show it (the lower of the two excesses is kept). Detection: the 50 kHz-binned power over its 20th-percentile floor
in a 10 MHz window, more than 8 dB, runs joined across gaps of up to 3 bins. Widths are underestimates for wide signals (the floor window eats the plateau: a 6 MHz TV channel reads as 1.5 MHz).
Two antennas, or two times of day, will differ: bursty emitters (Wi-Fi, BLE, intermittent radios) come and go, so `--compare` is a guide, and the 'in both' list is the believable part.
"""
import argparse
import collections
import os
import sys
import time

import numpy as np
from rpps.math import percentile_filter

import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); import _paths  # noqa: E402,F401
import radio_io

FS = 20e6; STEP = 16e6; NFFT = 2**14
ALLOCATIONS = [ # (lo MHz, hi MHz, name): North America, by the ranges seen in this work; anything else prints as 'other'
    (88, 108, "FM broadcast"), (108, 137, "aviation"), (144, 148, "2 m amateur"), (148, 174, "VHF land mobile / weather / marine"), (174, 216, "VHF TV"),
    (400, 470, "UHF land mobile / 70 cm"), (470, 698, "UHF TV (ATSC)"), (728, 756, "LTE b12/13/17 DL"), (756, 776, "700 MHz public safety"),
    (851, 869, "800 MHz public safety / SMR"), (869, 894, "cellular DL (b5)"), (902, 928, "ISM 900"), (1525, 1610, "mobile satellite / GNSS"),
    (1930, 1995, "PCS DL (b2)"), (2110, 2200, "AWS DL (b4/b66)"), (2400, 2484, "2.4 GHz Wi-Fi / BLE"), (3550, 3700, "CBRS"),
    (3700, 4000, "C-band / n77?"), (5150, 5925, "5 GHz Wi-Fi"), (5925, 6000, "U-NII-5 edge")]

def label(f):
    for lo, hi, n in ALLOCATIONS:
        if lo <= f < hi:
            return n
    return "other"

def sweep(sdr, lo, hi, gain=None):
    frs, dbs = [], []
    cf = lo + STEP/2
    while cf - STEP/2 < hi:
        sdr.set_rx_freq(cf, flush=True)
        if gain is not None:
            radio_io.set_gain(sdr, gain); time.sleep(0.03) # a fixed, lower gain: at the ladder's high gains the HackRF makes broad plateaus around its tuned frequency that follow the tuning
        else:
            for g in radio_io.LADDER:
                radio_io.set_gain(sdr, g); time.sleep(0.03)
                if radio_io.clipped(radio_io.grab(sdr, 2**18)) < 1e-4:
                    break
        x = radio_io.grab(sdr, 2**21)
        seg = x[:len(x)//NFFT*NFFT].reshape(-1, NFFT)*np.hanning(NFFT)
        P = np.mean(np.abs(np.fft.fftshift(np.fft.fft(seg, axis=1), axes=1))**2, 0)
        fr = np.fft.fftshift(np.fft.fftfreq(NFFT, 1/FS)) + cf
        keep = (np.abs(fr - cf) < STEP/2) & (np.abs(fr - cf) > 150e3)
        frs.append(fr[keep]); dbs.append(10*np.log10(P[keep] + 1e-20))
        cf += STEP
    return np.concatenate(frs), np.concatenate(dbs)

def do_sweep(prefix, lo, hi, gain=None):
    os.makedirs(os.path.dirname(os.path.abspath(prefix)), exist_ok=True)
    sdr = radio_io.open_radio(cf=100e6); sdr.start_rx()
    try:
        for k, shift in ((1, 0.0), (2, 8.0)):
            t0 = time.time(); frs, dbs = [], []
            a = lo
            while a < hi:
                b = min(a + 800, hi)
                fr, db = sweep(sdr, (a + shift)*1e6, (b + shift)*1e6, gain); frs.append(fr); dbs.append(db)
                a = b
                print(f"pass {k}: up to {b:.0f} MHz, {time.time() - t0:.0f} s", flush=True)
            np.savez(f"{prefix}_{k}.npz", fr=np.concatenate(frs), db=np.concatenate(dbs))
    finally:
        sdr.stop_rx(); sdr.close()

def binned(path, bw=50e3):
    d = np.load(path); fr, db = d["fr"], d["db"]
    edges = np.arange(5e6, 6.1e9, bw); idx = np.digitize(fr, edges) - 1; n = len(edges) - 1
    m = (idx >= 0) & (idx < n)
    pw = np.bincount(idx[m], 10**(db[m]/10), n); c = np.bincount(idx[m], minlength=n)
    ok = c > 0
    p = np.full(n, np.nan); p[ok] = 10*np.log10(pw[ok]/c[ok])
    ctr = edges[:-1] + bw/2
    p = np.interp(ctr, ctr[ok], p[ok])
    floor = percentile_filter(p, 20, size=int(10e6/bw), mode="nearest")
    return ctr, p - floor, ok

def confirmed(prefix, threshold=8.0):
    """Emitters present in both passes: list of (centre MHz, width MHz, excess dB)"""
    c, e1, o1 = binned(f"{prefix}_1.npz"); _, e2, o2 = binned(f"{prefix}_2.npz")
    ex = np.minimum(e1, e2); ex[~(o1 & o2)] = 0
    f = c/1e6; r = f % 20
    ex[np.minimum(r, 20 - r) < 0.13] = 0 # the HackRF's own spurs
    on = ex > threshold; out = []; i = 0; n = len(on)
    while i < n:
        if on[i]:
            j = i
            while j < n and (on[j] or on[j:j + 3].any()):
                j += 1
            out.append(((c[i] + c[j - 1])/2e6, (c[j - 1] - c[i])/1e6 + 0.05, float(ex[i:j].max())))
            i = j
        else:
            i += 1
    return out

def same(a, b):
    return abs(a[0] - b[0]) < max(0.15, (a[1] + b[1])/2)

def grouped(lst, top=5):
    g = collections.defaultdict(list)
    for e in lst:
        g[label(e[0])].append(e)
    for k, v in sorted(g.items(), key=lambda kv: min(e[0] for e in kv[1])):
        best = sorted(v, key=lambda t: -t[2])[:top]
        print(f"  {k:36s} n={len(v):3d}  " + ", ".join(f"{f:.3f} (w {w:.2f}, +{e:.0f})" for f, w, e in best))

def do_report(prefix, compare):
    now = confirmed(prefix)
    print(f"{len(now)} confirmed emitters in {prefix} (present in both passes)")
    if not compare:
        grouped(now); return
    old = confirmed(compare)
    both = [n for n in now if any(same(n, o) for o in old)]
    new = [n for n in now if not any(same(n, o) for o in old)]
    gone = [o for o in old if not any(same(n, o) for n in now)]
    print(f"{len(old)} in {compare}; in both {len(both)}, new {len(new)}, gone {len(gone)}")
    print("\nin both (the persistent signals):"); grouped(both)
    print("\nnew:"); grouped(new, 4)
    print("\ngone:"); grouped(gone, 4)

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sweep"); s.add_argument("prefix"); s.add_argument("--lo", type=float, default=30); s.add_argument("--hi", type=float, default=6000); s.add_argument("--gain", type=float, help="fixed gain (a fraction 0-1 of the radio's range) instead of the highest non-clipping rung (use a low one where tune-following plateaus show up)")
    r = sub.add_parser("report"); r.add_argument("prefix"); r.add_argument("--compare")
    a = ap.parse_args(argv)
    if a.cmd == "sweep":
        do_sweep(a.prefix, a.lo, a.hi, a.gain)
    else:
        do_report(a.prefix, a.compare)

if __name__ == "__main__":
    main()
