"""Scan the TV channels for ATSC (8VSB) stations, pick the best gain for each, decode the ones that lock and list their programs

    python projects/atsc/atsc_scan.py                       # UHF 14-36 and VHF 7-13, writes data/atsc/ch<N>.ts
    python projects/atsc/atsc_scan.py --channels 15 19 35 --seconds 4 --out-dir data/atsc

A channel n spans 470 + 6 x (n - 14) MHz (UHF 14-36) or 174 + 6 x (n - 7) MHz (VHF 7-13), 6 MHz wide, pilot 0.31 MHz above the lower edge. The tune centre is 9 MHz above the lower edge, so the
HackRF's DC spike falls outside the channel. Per channel: a 0.4 s capture at each of a few gain settings, the equalized MER (atsc_probe.recover) of each, the best one (clipping rejected)
decides the gain; a channel that does not lock at any is reported as no station. Locked channels are captured for --seconds, decoded with the block-adaptive equalizer of atsc_probe (the channel moves during a capture) and
summarised with ts_info.py (the PSIP virtual channel names). Meant to be upstreamed into RPPS.
"""
import argparse
import os
import sys
import time

import numpy as np

import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); import _paths  # noqa: E402,F401
import atsc_decode
import atsc_probe as ap
import radio_io
import ts_info

FS = 20e6
GAINS = [(1, 24, 24), (1, 32, 16), (0, 32, 24), (1, 16, 32), (1, 40, 8)]


def lower_edge(n):
    return 470.0 + 6*(n - 14) if n >= 14 else 174.0 + 6*(n - 7)


def main(argv=None):
    ap_ = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap_.add_argument("--channels", type=int, nargs="+", default=list(range(7, 14)) + list(range(14, 37)))
    ap_.add_argument("--seconds", type=float, default=3.0, help="capture length of a locked channel"); ap_.add_argument("--out-dir", default="data/atsc"); ap_.add_argument("--keep", action="store_true", help="keep the 3 s captures (480 MB each)")
    a = ap_.parse_args(argv)
    os.makedirs(a.out_dir, exist_ok=True)
    sdr = radio_io.open_radio(cf=500e6, fs=FS); sdr.start_rx()
    found = []
    try:
        for n in a.channels:
            lo = lower_edge(n); cf = (lo + 9)*1e6
            sdr.set_rx_freq(cf, flush=True)
            best = None
            for g in GAINS:
                radio_io.set_gain(sdr, g); time.sleep(0.05)
                x = radio_io.grab(sdr, int(FS*0.4)); clip = radio_io.clipped(x)
                if clip > 1e-4:
                    continue
                try:
                    r = ap.recover(x.astype(np.complex64), FS, cf, lo*1e6, verbose=False)
                except Exception as e:
                    print(f"channel {n:2d}: recover failed at gain {g}: {type(e).__name__}: {str(e)[:80]}", file=sys.stderr, flush=True)
                    continue
                if r.fs_seg is not None and (best is None or r.mer_eq > best[0]):
                    best = (r.mer_eq, g)
                if r.fs_seg is None and g == GAINS[0]:
                    break # no field sync at the first setting: not a station (or too weak for any gain)
            if best is None:
                print(f"channel {n:2d} ({lo:g}-{lo + 6:g} MHz): no ATSC lock", flush=True); continue
            mer, g = best
            radio_io.set_gain(sdr, g); time.sleep(0.1)
            x = radio_io.grab(sdr, int(FS*a.seconds)).astype(np.complex64)
            path = os.path.join(a.out_dir, f"ch{n}.cf64"); x.tofile(path)
            print(f"channel {n:2d} ({lo:g}-{lo + 6:g} MHz): locked, gain {radio_io.describe(sdr)}, equalized MER {mer:.1f} dB; decoding {a.seconds:g} s ...", flush=True)
            found.append((n, lo, mer, g, path))
    finally:
        sdr.stop_rx(); sdr.close()
    for n, lo, mer, g, path in found:
        print(f"\n=== channel {n} ({lo:g}-{lo + 6:g} MHz, MER {mer:.1f} dB)")
        xx = np.fromfile(path, np.complex64)
        r = ap.recover(xx, FS, (lo + 9)*1e6, lo*1e6, verbose=False)
        tss = []
        if r.fs_seg is not None:
            import contextlib, io
            with contextlib.redirect_stdout(io.StringIO()):
                tss.append(atsc_decode.ts_from_soft(r.soft, r.fs_seg))
        del xx
        if not a.keep:
            os.remove(path)
        if not tss:
            print("  no decodable piece"); continue
        ts = np.concatenate(tss); out = os.path.join(a.out_dir, f"ch{n}.ts"); ts.tofile(out)
        print(f"  {len(ts)} transport packets -> {out}")
        ts_info.main([out])
    return found


if __name__ == "__main__":
    main()
