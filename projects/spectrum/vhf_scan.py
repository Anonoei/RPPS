"""VHF / UHF narrowband channel scanner (108-470 MHz): which channels are in use, how busy, AM or FM, and what service they belong to

    python projects/spectrum/vhf_scan.py                       # 108-470 MHz, 3 s per 16 MHz tile
    python projects/spectrum/vhf_scan.py --lo 118 --hi 137 --seconds 10     # aviation voice band, longer
    python projects/spectrum/vhf_scan.py --lo 144 --hi 148 --keep captures/ # amateur 2 m; keep the tile captures
    python projects/spectrum/vhf_scan.py --files "captures/vhf_{}.cf64" --lo 144 --hi 160 --cf 152   # analyse a saved tile

For each 16 MHz tile (20 MS/s, the HackRF's DC spike masked, tile edges dropped) the gain is the highest on a ladder that does not clip. The capture is turned into a spectrogram with
12.5 kHz channel bins and 10 ms frames; per channel bin: the noise floor is the median power over all bins and frames of the tile, a frame is *active* when the bin is more than 8 dB above it,
duty = fraction of active frames, level = the strongest excess. Adjacent active bins are merged into one channel (voice channels are 12.5-25 kHz wide) and its centre refined.
AM or FM: for the strongest channels the channel is extracted (mix, 16 kHz low-pass) and the envelope's coefficient of variation during active frames is measured: constant-envelope
modulations (FM, FSK, GMSK, C4FM) give ~0.05, voice AM (aviation, 118-137 and 225-400 MHz) gives 0.3 or more.
The HackRF's own spurs at multiples of 20 MHz are removed. Labels come from the allocation and channel tables in this file (North America: FCC / ITU-R M.1084 marine channels, NOAA weather radio, FRS/GMRS/MURS,
aviation band plan): a label says what the frequency is *for*, not what was decoded. Nothing is demodulated for listening or decoded here: only occupancy and modulation type.
"""
import argparse
import os
import sys
import time
from collections import defaultdict

import numpy as np
from scipy.signal import resample_poly

import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); import _paths  # noqa: E402,F401
import radio_io

FS = 20e6; TILE = 16e6; BIN = 12.5e3

SERVICES = [ # (lo MHz, hi MHz, name): coarse allocations
    (108.0, 117.975, "aviation navigation (VOR / ILS localizer)"), (117.975, 137.0, "aviation voice (AM, 25 kHz)"), (137.0, 138.0, "weather satellites / space ops"),
    (138.0, 144.0, "federal / military"), (144.0, 148.0, "amateur 2 m"), (148.0, 150.8, "land mobile / federal"), (150.8, 156.0, "land mobile (business, public safety, MURS)"),
    (156.0, 162.0125, "marine VHF"), (162.0125, 162.4, "federal / marine AIS 2"), (162.4, 162.55, "NOAA weather radio"), (162.55, 174.0, "federal land mobile / public safety"),
    (174.0, 216.0, "TV ch 7-13"), (216.0, 220.0, "marine / telemetry"), (220.0, 222.0, "land mobile"), (222.0, 225.0, "amateur 1.25 m"), (225.0, 400.0, "military air band (AM)"),
    (400.0, 406.0, "meteorological / space"), (406.0, 406.1, "emergency beacons (ELT / EPIRB)"), (406.1, 420.0, "federal"), (420.0, 450.0, "amateur 70 cm / radiolocation"),
    (450.0, 460.0, "land mobile (business, public safety, repeaters)"), (460.0, 470.0, "land mobile / FRS / GMRS repeaters")]
MARINE = {} # channel number -> ship transmit MHz (ITU-R M.1084)
for n in range(1, 29): MARINE[n] = 156.05 + 0.05*(n - 1)
for n in range(60, 89): MARINE[n] = 156.025 + 0.05*(n - 60)
NOAA = {1: 162.400, 2: 162.425, 3: 162.450, 4: 162.475, 5: 162.500, 6: 162.525, 7: 162.550}
FRS_GMRS = [462.5625, 462.5875, 462.6125, 462.6375, 462.6625, 462.6875, 462.7125, 467.5625, 467.5875, 467.6125, 467.6375, 467.6625, 467.6875, 467.7125, 462.5500, 462.5750, 462.6000,
            462.6250, 462.6500, 462.6750, 462.7000, 462.7250]
MURS = [151.820, 151.880, 151.940, 154.570, 154.600]
NAMED = [(121.5, "aviation emergency (guard)"), (243.0, "military emergency (guard)"), (123.45, "air-to-air"), (122.75, "air-to-air"), (161.975, "AIS 1 (marine, 9600 bd GMSK)"),
         (162.025, "AIS 2 (marine, 9600 bd GMSK)"), (156.8, "marine ch 16 (distress / calling)"), (131.55, "ACARS"), (131.725, "ACARS"), (131.825, "ACARS"), (136.975, "VDL mode 2"),
         (144.39, "APRS (amateur, 1200 bd AFSK)"), (146.52, "amateur 2 m national simplex"), (446.0, "amateur 70 cm national simplex"), (162.55, "NOAA weather"),
         (406.025, "emergency beacon"), (156.525, "marine ch 70 (DSC digital selective calling)")]

def label(f):
    for hz, n in NAMED:
        if abs(f - hz) < 0.0075:
            return n
    for k, hz in NOAA.items():
        if abs(f - hz) < 0.0075:
            return f"NOAA weather radio ch {k}"
    for hz in FRS_GMRS:
        if abs(f - hz) < 0.0075:
            return "FRS / GMRS"
    for hz in MURS:
        if abs(f - hz) < 0.0075:
            return "MURS"
    if 156.0 <= f < 162.0:
        for n, hz in MARINE.items():
            if abs(f - hz) < 0.0075 and n < 80:
                return f"marine ch {n}"
    for lo, hi, n in SERVICES:
        if lo <= f < hi:
            return n
    return "other"

def tiles(lo, hi):
    out = []; half = TILE/2/1e6 # MHz
    cf = lo + half
    while cf - half < hi:
        out.append(cf*1e6); cf += TILE/1e6
    return out

def spectrogram(x, cf, frame_s=0.01):
    """Channel-bin power over time: returns (freqs Hz, array (frames, bins)). 12.5 kHz bins: a 1600-point FFT at 20 MS/s; 125 FFTs averaged per 10 ms frame."""
    nfft = int(FS/BIN)
    per = int(frame_s*FS/nfft)
    n_fft = len(x)//nfft
    X = x[:n_fft*nfft].reshape(n_fft, nfft)*np.hanning(nfft)
    P = np.abs(np.fft.fftshift(np.fft.fft(X, axis=1), axes=1))**2
    n_fr = n_fft//per
    P = P[:n_fr*per].reshape(n_fr, per, nfft).mean(1)
    f = cf + np.fft.fftshift(np.fft.fftfreq(nfft, 1/FS))
    return f, P

def channels_in_tile(x, cf, lo, hi, thresh_db=8.0):
    """Active channels in one tile: list of dicts(freq MHz, duty, level dB, width kHz)"""
    f, P = spectrogram(x, cf)
    db = 10*np.log10(P + 1e-20)
    floor = np.median(db)
    ex = db - floor
    keep = (np.abs(f - cf) < TILE/2) & (np.abs(f - cf) > 120e3) & (f >= lo*1e6) & (f < hi*1e6)
    # spurs at multiples of 20 MHz (HackRF's own)
    mhz = f/1e6; r = mhz % 20
    keep &= ~(np.minimum(r, 20 - r) < 0.08)
    active = ex > thresh_db
    duty = active.mean(0)
    peak = ex.max(0)
    sel = keep & ((duty >= 0.02) | (peak >= thresh_db + 6))
    out = []; i = 0; idx = np.where(sel)[0]
    while i < len(idx):
        j = i
        while j + 1 < len(idx) and idx[j + 1] - idx[j] <= 1:
            j += 1
        bins = idx[i:j + 1]
        w = P[:, bins].mean(0)
        centre = float(np.sum(f[bins]*w)/np.sum(w))
        k = bins[int(np.argmax(duty[bins]*100 + peak[bins]))]
        out.append(dict(freq=centre/1e6, duty=float(duty[bins].max()), level=float(peak[bins].max()), width=(len(bins))*BIN/1e3, frames=active[:, bins].any(1)))
        i = j + 1
    return out

def modulation(x, cf, freq, frames_active, frame_s=0.01):
    """'AM' / 'FM-like' / '?' from the envelope of the channel during its active frames"""
    if frames_active.sum() < 3:
        return "?", 0.0
    t = np.arange(len(x))
    y = x*np.exp(-2j*np.pi*(freq*1e6 - cf)*t/FS)
    z = resample_poly(y, 1, 625) # 20 MS/s -> 32 kS/s: a ~16 kHz wide channel
    fs2 = 32e3
    fr = int(frame_s*fs2)
    n = len(z)//fr
    env = np.abs(z[:n*fr]).reshape(n, fr)
    act = frames_active[:n]
    if act.sum() < 3:
        return "?", 0.0
    v = env[act].reshape(-1)
    cv = float(np.std(v)/np.mean(v))
    return ("AM" if cv > 0.25 else "FM / constant envelope"), cv

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lo", type=float, default=108.0); ap.add_argument("--hi", type=float, default=470.0)
    ap.add_argument("--seconds", type=float, default=3.0, help="dwell per tile")
    ap.add_argument("--keep", help="save each tile's capture here as vhf_<centre MHz>.cf64")
    ap.add_argument("--files", help="analyse saved tiles (pattern with {} for the centre in MHz, e.g. captures/vhf_{}.cf64); needs --lo --hi to match the tiles")
    ap.add_argument("--cf", type=float, action="append", help="with --files: tile centre(s) in MHz")
    ap.add_argument("--min-duty", type=float, default=0.02, help="list channels active at least this fraction of the time (default 0.02)")
    a = ap.parse_args(argv)
    sdr = None
    if not a.files:
        sdr = radio_io.open_radio(cf=100e6); radio_io.start_rx(sdr)
    found = []
    try:
        cfs = [c*1e6 for c in a.cf] if a.files else tiles(a.lo, a.hi)
        for cf in cfs:
            if a.files:
                x = np.fromfile(a.files.format(f"{cf/1e6:g}"), np.complex64).astype(np.complex128); g = None; clip = 0.0
            else:
                sdr.set_rx_freq(cf, flush=True)
                x, g, clip = radio_io.capture_clean(sdr, FS, a.seconds, check_seconds=0.5)
                x = x.astype(np.complex64)
                if a.keep:
                    x.tofile(os.path.join(a.keep, f"vhf_{cf/1e6:g}.cf64"))
                x = x.astype(np.complex128)
            chs = channels_in_tile(x, cf, a.lo, a.hi)
            for c in chs:
                if c["duty"] >= a.min_duty or c["level"] >= 14:
                    c["mod"], c["cv"] = modulation(x, cf, c["freq"], c["frames"])
                    c["label"] = label(c["freq"]); c["tile"] = cf/1e6; found.append(c)
            print(f"tile {cf/1e6:6.1f} MHz: gain {g}, clipped {clip:.0e}, {len(chs)} active channels", flush=True)
    finally:
        if sdr:
            sdr.stop_rx(); sdr.close()
    by = defaultdict(list)
    for c in found:
        key = c["label"] if c["label"].split(" ch ")[0] in ("marine", "NOAA weather radio") or c["label"] in ("FRS / GMRS", "MURS") else next((n for lo, hi, n in SERVICES if lo <= c["freq"] < hi), "other")
        by[key].append(c)
    print(f"\n{len(found)} active channels, grouped by allocation (duty = share of 10 ms frames above the floor, level = strongest excess over the tile floor, mod = envelope test):")
    order = sorted(by, key=lambda k: min(c["freq"] for c in by[k]))
    for k in order:
        print(f"\n{k}:")
        for c in sorted(by[k], key=lambda c: c["freq"]):
            extra = "" if c["label"] == k else f"  [{c['label']}]"
            print(f"  {c['freq']:9.4f} MHz  duty {100*c['duty']:5.1f}%  level +{c['level']:4.1f} dB  width {c['width']:5.1f} kHz  {c['mod']} (cv {c['cv']:.2f}){extra}")
    return found

if __name__ == "__main__":
    main()
