"""Choose the receive gain (a fraction 0-1 of the radio's range, on any radio) that gets the most signal with the fewest artifacts

    python projects/common/gain_search.py 507 --sig 500.5 505.5 --vac 507.5 512     # tune at 507 MHz, signal 500.5-505.5, empty 507.5-512
    python projects/common/gain_search.py 98 --fs 2.4 --sig 97.4 97.6 --vac 98.2 98.6

Every step of the gain range is captured and scored. What counts against a setting:
  clipping       the ADC saturates (strong signals overload the front end): a large penalty
  artifact lines a line in the empty band that most settings don't show. A line seen at (nearly) every setting is outside
                 the receiver's control (a real carrier or the radio's own spur, e.g. multiples of 20 MHz); one that only
                 appears at some gains is overload or intermodulation the gain made
  weak signal    SNR is what is left to maximise
Use `search()` from other scripts: the answer is the gain fraction and the table of what was tried. Any pysdrlib radio (see radio_io.py); with $RPPS_CHANNELS=2 on a B210
(the same signal on both inputs) two settings are measured per capture.
"""
import argparse
import os
import sys
import time
from types import SimpleNamespace

import numpy as np

import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); import _paths  # noqa: E402,F401
import radio_io

GAINS = tuple(round(g, 2) for g in np.linspace(0, 1, 21)) # gain fractions tried (about 4 dB apart on a HackRF, 3.8 on a B210)
CLIP = 0.98 # |I| or |Q| above this is saturated
NFFT = 2**14

def _spectrum(x, Fs, cf):
    seg = x[:len(x)//NFFT*NFFT].reshape(-1, NFFT)*np.hanning(NFFT)
    P = 10*np.log10(np.mean(np.abs(np.fft.fftshift(np.fft.fft(seg, axis=1), axes=1))**2, 0) + 1e-20)
    return np.fft.fftshift(np.fft.fftfreq(NFFT, 1/Fs)) + cf, P

def _lines(fr, P, mask, over=6.0, merge=30e3):
    """(freq, dB over the median) of each line in the masked part of the spectrum"""
    med = np.median(P[mask])
    out = []
    for i in np.where(mask & (P > med + over))[0]:
        if out and fr[i] - out[-1][0] < merge:
            if P[i] > out[-1][2]:
                out[-1] = (fr[i], P[i] - med, P[i])
            continue
        out.append((fr[i], P[i] - med, P[i]))
    return [(f, h) for f, h, _ in out]

def measure(sdr, cf, Fs, sig, vac, samples=2**21):
    """Capture at the current gain; returns SNR, clip fraction, rms, and the lines in the empty band"""
    return measure_x(radio_io.grab(sdr, samples), cf, Fs, sig, vac)

def measure_x(x, cf, Fs, sig, vac):
    """The same measurements of samples already captured"""
    fr, P = _spectrum(x, Fs, cf)
    dc = np.abs(fr - cf) < 150e3 # The DC spike is at the tuned frequency: not part of the signal or the floor
    ms = (fr >= sig[0]) & (fr < sig[1]) & ~dc
    mv = (fr >= vac[0]) & (fr < vac[1]) & ~dc
    ps, pv = 10**(P[ms]/10), 10**(P[mv]/10)
    snr = 10*np.log10(max(ps.mean() - pv.mean(), 1e-12)/pv.mean())
    clip = float(np.mean((np.abs(x.real) > CLIP) | (np.abs(x.imag) > CLIP)))
    return SimpleNamespace(snr=float(snr), clip=clip, rms=float(np.std(x)), lines=_lines(fr, P, mv))

def search(sdr, cf, Fs, sig, vac, gains=GAINS, verbose=True):
    """Try every gain in `gains` (fractions 0-1). sdr is an open, receiving radio at cf, Fs. sig/vac: (lo, hi) in Hz.
    Returns (gain, rows); each row is a namespace with the setting (`gain`), the measurements and the score."""
    rows = []
    settings = list(gains)
    step = 2 if radio_io.dual(sdr) else 1 # two channels: two settings per capture
    for k in range(0, len(settings), step):
        group = settings[k:k + step]
        for ch, st in enumerate(group):
            radio_io.set_gain(sdr, st, channel=ch)
        time.sleep(0.04)
        if step == 2:
            x = radio_io.grab2(sdr, 2**21)
            got = [measure_x(x[ch], cf, Fs, sig, vac) for ch in range(len(group))]
        else:
            got = [measure(sdr, cf, Fs, sig, vac)]
        for r, st in zip(got, group):
            r.setting = st
            rows.append(r)
    # A line present at most unclipped settings is the band's own (or the radio's); the rest are what the gain caused
    ok = [r for r in rows if r.clip < 1e-4]
    common = []
    for f, _ in (l for r in ok for l in r.lines):
        if not any(abs(f - c) < 60e3 for c in common) and sum(any(abs(f - g) < 60e3 for g, _ in r.lines) for r in ok) >= 0.3*len(ok): # (a real line only clears the threshold at the higher gains)
            common.append(f)
    for r in rows:
        r.artifacts = [(f, h) for f, h in r.lines if not any(abs(f - c) < 60e3 for c in common)]
        r.score = r.snr - (50 if r.clip > 1e-4 else 0) - sum(2 + h/2 for _, h in r.artifacts)
        if r.rms < 0.15:
            r.score -= 2 # Leaves the ADC's range unused: its own noise and quantisation start to count
        if verbose:
            print(f"gain {r.setting:.2f} | clip {r.clip:.0e} rms {r.rms:.2f} | SNR {r.snr:5.1f} dB | "
                  f"artifacts {len(r.artifacts)} | score {r.score:5.1f}", flush=True)
    best = max(rows, key=lambda r: r.score)
    best.common_lines = common
    return best.setting, rows

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("freq", type=float, help="Tune to this [MHz]")
    ap.add_argument("--fs", type=float, default=20.0, help="Sample rate [MS/s] (default 20)")
    ap.add_argument("--sig", type=float, nargs=2, required=True, metavar=("LO", "HI"), help="The signal's band [MHz]")
    ap.add_argument("--vac", type=float, nargs=2, required=True, metavar=("LO", "HI"), help="An empty band near it [MHz]")
    ap.add_argument("--quiet", action="store_true", help="Only print the answer")
    a = ap.parse_args()
    cf, Fs = a.freq*1e6, a.fs*1e6
    sdr = radio_io.open_radio(cf=cf, fs=Fs); radio_io.start_rx(sdr)
    try:
        gain, rows = search(sdr, cf, Fs, [v*1e6 for v in a.sig], [v*1e6 for v in a.vac], verbose=not a.quiet)
    finally:
        sdr.stop_rx(); sdr.close()
    best = max(rows, key=lambda r: r.score)
    print(f"\nBest: gain {gain:.2f}: SNR {best.snr:.1f} dB, rms {best.rms:.2f}, clip {best.clip:.0e}, "
          f"{len(best.artifacts)} artifact lines")
    if best.common_lines:
        print("Lines present at most settings (not the gain's doing): " + ", ".join(f"{f/1e6:.3f} MHz" for f in best.common_lines))
    print(f"  flag: --gain {gain:.2f}")

if __name__ == "__main__":
    sys.exit(main())
