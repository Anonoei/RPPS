"""Channel occupancy of one capture: which narrowband channels are active, how often, and which are probably HackRF spurs

    python projects/spectrum/channel_watch.py data/avi/avi_125.cf64 --fs 8 --cf 125 --lo 121 --hi 129          # a saved cf64 capture (MHz, MS/s)
    python projects/spectrum/channel_watch.py --tune 163 --fs 4 --seconds 30 --gain 0.67 --keep data/ais/ais_163.cf64
    python projects/spectrum/channel_watch.py data/ais/ais_163.cf64 --fs 4 --cf 163 --at 161.975 162.025 162.55   # duty at named frequencies only

Spectrogram with `--bin` kHz channels and `--frame` ms frames; a frame is *active* in a channel when it is more than `--thresh` dB above the capture's median. Per channel: duty (share of
active frames) and the strongest excess. Adjacent channels are merged. `spur?` marks lines within 3 kHz of a whole MHz: the HackRF's own spurs sit on multiples of its sample and clock
rates, while a real transmitter is a clock-error away (about +15 ppm, so +2 kHz at 140 MHz) from its nominal frequency. Combs of evenly spaced 100 %-duty lines (2.35 MHz apart at 20 MS/s, 0.6 MHz at
8 MS/s) are an artifact whose spacing changes with the sample rate; change the sample rate and see which lines move. Pick the gain first with gain_search.py.
For the details of one bursty signal use burst_tones.py. Meant to be upstreamed into RPPS.
"""
import argparse
import os
import sys
import time

import numpy as np

import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); import _paths  # noqa: E402,F401
import radio_io


def spectrogram(x, fs, cf, bin_hz, frame_s):
    """(freqs Hz, power array (frames, bins)): FFT size fs/bin_hz, averaged to frame_s frames"""
    nfft = int(round(fs/bin_hz)); n = len(x)//nfft
    X = x[:n*nfft].reshape(n, nfft)*np.hanning(nfft)
    P = np.abs(np.fft.fftshift(np.fft.fft(X, axis=1), axes=1))**2
    per = max(1, int(frame_s*fs/nfft)); m = n//per
    return cf + np.fft.fftshift(np.fft.fftfreq(nfft, 1/fs)), P[:m*per].reshape(m, per, nfft).mean(1)


def channels(x, fs, cf, lo, hi, bin_hz=12.5e3, frame_s=0.01, thresh=8.0, min_duty=0.005, min_peak=14.0, dc_hz=100e3):
    """Active channels: list of dicts(freq MHz, duty, peak dB, width kHz, spur)"""
    f, P = spectrogram(x, fs, cf, bin_hz, frame_s)
    ex = 10*np.log10(P + 1e-20); ex -= np.median(ex)
    duty = (ex > thresh).mean(0); peak = ex.max(0)
    sel = np.where((abs(f - cf) > dc_hz) & (f >= lo) & (f <= hi) & ((duty >= min_duty) | (peak >= min_peak)))[0]
    out = []; i = 0
    while i < len(sel):
        j = i
        while j + 1 < len(sel) and sel[j + 1] - sel[j] <= 2:
            j += 1
        b = sel[i:j + 1]; k = b[np.argmax(peak[b])]
        mhz = f[k]/1e6
        out.append(dict(freq=mhz, duty=float(duty[b].max()), peak=float(peak[b].max()), width=len(b)*bin_hz/1e3, spur=abs(mhz - round(mhz)) < 0.003))
        i = j + 1
    return out, f, ex


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file", nargs="?", help="cf64 capture (omit with --tune)")
    ap.add_argument("--fs", type=float, required=True, help="sample rate [MS/s]")
    ap.add_argument("--cf", type=float, help="capture centre [MHz] (file mode)")
    ap.add_argument("--tune", type=float, help="capture live at this centre [MHz]")
    ap.add_argument("--seconds", type=float, default=10.0)
    radio_io.add_gain_arg(ap, 0.67)
    ap.add_argument("--keep", help="save the live capture here (cf64)")
    ap.add_argument("--lo", type=float, help="lowest frequency to list [MHz] (default: whole capture)")
    ap.add_argument("--hi", type=float)
    ap.add_argument("--bin", type=float, default=12.5, help="channel width [kHz]")
    ap.add_argument("--frame", type=float, default=10.0, help="frame length [ms]")
    ap.add_argument("--thresh", type=float, default=8.0, help="active when this many dB over the median")
    ap.add_argument("--min-duty", type=float, default=0.005)
    ap.add_argument("--at", type=float, nargs="+", help="report duty only at these frequencies [MHz] (+-10 kHz)")
    a = ap.parse_args(argv)
    fs = a.fs*1e6
    if a.tune is not None:
        cf = a.tune*1e6
        sdr = radio_io.open_radio(cf=cf, fs=fs); radio_io.set_gain(sdr, a.gain); sdr.start_rx(); time.sleep(0.2)
        try:
            x = radio_io.grab(sdr, int(fs*a.seconds))
        finally:
            sdr.stop_rx(); sdr.close()
        print(f"captured {a.seconds:g} s at {a.tune:g} MHz, {a.fs:g} MS/s, clipped {radio_io.clipped(x):.1e}")
        if a.keep:
            x.astype(np.complex64).tofile(a.keep)
    else:
        if not a.file or a.cf is None:
            ap.error("file mode needs FILE and --cf")
        cf = a.cf*1e6; x = np.fromfile(a.file, np.complex64)
    x = x.astype(np.complex64)
    lo = (a.lo if a.lo is not None else cf/1e6 - a.fs/2)*1e6; hi = (a.hi if a.hi is not None else cf/1e6 + a.fs/2)*1e6
    chs, f, ex = channels(x, fs, cf, lo, hi, a.bin*1e3, a.frame/1e3, a.thresh, a.min_duty)
    if a.at:
        for mhz in a.at:
            j = np.argmin(abs(f - mhz*1e6)); w = ex[:, max(0, j - 1):j + 2].max(1)
            print(f"{mhz:9.4f} MHz: median {np.median(w):+5.1f} dB, p99 {np.percentile(w, 99):+5.1f}, max {w.max():+5.1f}, active {100*(w > a.thresh).mean():5.2f}% of frames")
        return chs
    print(f"{len(chs)} active channels ({a.bin:g} kHz, {a.frame:g} ms frames, > {a.thresh:g} dB over median):")
    for c in chs:
        print(f"  {c['freq']:9.4f} MHz  duty {100*c['duty']:5.1f}%  peak +{c['peak']:4.1f} dB  width {c['width']:5.1f} kHz" + ("  spur? (whole MHz)" if c["spur"] else ""))
    return chs


if __name__ == "__main__":
    main()
