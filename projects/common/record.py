"""Record a capture (any pysdrlib radio, see radio_io.py) to a complex64 file

    python projects/common/record.py 507 20 0.6 data/atsc19.cf32 --gain 0.8
    python projects/common/record.py 104.8 2.4 8 data/fm.cf32            # MHz, MS/s, seconds, file

Pick the gain with gain_search.py first: the right setting depends on the band and on what else is on the air.
Read it back with rp.source.File("cf32", path, Fs=..., cf=...).
"""
import argparse
import os
import sys

import numpy as np

import rpps as rp
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); import _paths  # noqa: E402,F401
import radio_io

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("freq", type=float, help="Centre frequency [MHz]")
    ap.add_argument("fs", type=float, help="Sample rate [MS/s]")
    ap.add_argument("seconds", type=float)
    ap.add_argument("path")
    radio_io.add_gain_arg(ap)
    a = ap.parse_args()
    sdr = radio_io.open_radio(cf=a.freq*1e6, fs=a.fs*1e6)
    radio_io.set_gain(sdr, a.gain)
    sdr.start_rx()
    n, want, peak, clipped, lost = 0, int(a.fs*1e6*a.seconds), 0.0, 0, 0
    with rp.source.Sink("cf32", a.path) as sink:
        while n < want:
            b, m = sdr.read(min(2**17, want - n), timeout=3.0, meta=True)
            lost += m.dropped # samples lost before or inside this block (pysdrlib also warns as it happens)
            peak = max(peak, float(np.abs(b).max()))
            clipped += int(np.sum((np.abs(b.real) > 0.98) | (np.abs(b.imag) > 0.98)))
            sink.write(b)
            n += len(b)
    fs_now, cf_now = sdr.get_rx_sample_rate(), sdr.get_rx_freq()  # read before closing: on a remote radio these are calls to the server
    sdr.stop_rx(); sdr.close()
    print(f"{n} samples ({n/(a.fs*1e6):.2f} s), peak {peak:.2f}, clipped {clipped/n:.1e}, Fs {fs_now/1e6:g} MS/s, cf {cf_now/1e6:.4f} MHz -> {a.path}" + (f"  [LOST {lost} samples, ~{lost/(a.fs*1e6):.2f} s]" if lost else ""))

if __name__ == "__main__":
    sys.exit(main())
