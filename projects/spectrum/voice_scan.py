"""Scan VHF / UHF for voice: step across a range, find the channels that carry something, and say what each is (analog FM voice, AM voice, P25, DMR, other)

    python projects/spectrum/voice_scan.py --wav-dir data/voice                      # the default bands (30-88, 118-512, 758-776, 851-870 MHz), gain tuned per band
    python projects/spectrum/voice_scan.py --bands 450-470 --seconds 20 --no-tune

Per band the gain is tuned once with gain_search.search (the strongest line of a probe window is the signal, the quietest stretch the empty band); in a window where the tuned gain clips it backs off with
radio_io.capture_clean. Per window (8 MS/s, 7 MHz stepped): channel_watch.channels() lists the active 12.5 kHz channels (whole-MHz spurs dropped),
and each of the strongest --max channels is classified:
  P25 / DMR   : the frame sync of p25_sync.py / dmr_sync.py matches (at least 3 times)
  FM voice    : FM audio with most energy at 300-3000 Hz and a syllabic (2-8 Hz) envelope, see fm_audio.features
  AM voice    : the same test on the AM envelope (aviation and military bands are AM)
  carrier     : steady (> 95 % active) with no speech: also where the HackRF's combs and spurs land. Treat as unresolved, not as a signal
  other       : bursty or modulated, but neither speech nor a known digital mode
A channel that is *encrypted* is still labelled P25 / DMR here: check it with p25_voice.py / dmr_sync.py --timeline. Voice found is written as 16 kS/s WAV files into --wav-dir (local, untracked data/)
for you to listen to; nothing is uploaded. A 10 s window sees a channel only if someone transmits in it, so run it again at different times. Meant to be upstreamed into RPPS.
"""
import argparse
import os
import sys
import time
import wave

import numpy as np

import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); import _paths  # noqa: E402,F401
import channel_watch
import dmr_sync
import fm_audio
import gain_search
import radio_io
import p25_sync

FS = 8e6; STEP = 7e6


def digital(x, fs, cf, freq):
    """'P25', 'DMR' or None: frame-sync matches on the channel (at least 3 hits)"""
    sn, _ = p25_sync.symbols(x, fs, cf, freq)
    if len(p25_sync.sync_hits(sn)[0]) >= 3:
        return "P25"
    for P in dmr_sync.SYNCS.values():
        if len(p25_sync.sync_hits(sn, 0.9, P)[0]) >= 3:
            return "DMR"
    return None


def classify(x, fs, cf, freq):
    """(label, details dict, audio or None) for the channel at freq [Hz]. Digital sync is tested first: 4800-baud 4FSK audio has plenty of 300-3000 Hz energy and burst-rate envelope"""
    z, fs1 = fm_audio.baseband(x, fs, cf, freq)
    audio, mask, dev = fm_audio.fm_demod(z, fs1)
    ft = fm_audio.features(audio, mask)
    if ft is None:
        return "other", {}, None
    m = np.repeat(mask, int(round(fs1/16000)))[:len(dev)]; m = np.pad(m, (0, len(dev) - len(m)))
    ft["dev"] = float(np.std(dev[m])) # while the carrier is present only
    if ft["dev"] < 4e3:
        d = digital(x, fs, cf, freq)
        if d:
            return d, ft, None
    if ft["syl"] >= 1.5 and ft["voice"] >= 55 and ft["frac"] < 0.97:
        return "FM voice", ft, (audio, mask)
    am, m2 = fm_audio.am_demod(z, fs1)
    fa = fm_audio.features(am, m2)
    if fa and fa["syl"] >= 1.5 and fa["voice"] >= 55 and fa["frac"] < 0.97:
        return "AM voice", fa, (am - np.mean(am[m2]), m2)
    return ("carrier" if ft["frac"] > 0.95 else "other"), ft, None


def write_wav(path, audio, mask):
    y = np.where(mask, audio, 0.0); y = y/max(1e-9, np.abs(y).max())*0.8
    with wave.open(path, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000); w.writeframes((y*32767).astype(np.int16).tobytes())


BANDS = [(30, 88), (118, 174), (174, 230), (230, 400), (400, 440), (440, 512), (758, 776), (851, 870)] # MHz: low-band VHF, aviation + VHF land mobile / marine / amateur, TV 7-13, military UHF, UHF land mobile, T-band, 700 / 800 MHz public safety


def tune_band(sdr, lo, hi, seconds=2.0):
    """The band's gain by gain_search.search: probe the middle of the band, take its strongest line as the signal and its quietest stretch as the empty band. None when the probe sees nothing"""
    cf = (lo + hi)/2*1e6 if hi - lo < 8 else (lo + 4)*1e6 + (hi - lo - 8)/2*1e6
    sdr.set_rx_freq(cf, flush=True)
    x, g, _ = radio_io.capture_clean(sdr, FS, seconds, check_seconds=0.3); x = x.astype(np.complex64)
    chs, _, _ = channel_watch.channels(x, FS, cf, cf - STEP/2, cf + STEP/2, 12.5e3, 0.02, 8.0, 0.01, 10.0)
    chs = [c for c in chs if not c["spur"] and abs(c["freq"]*1e6 - cf) > 400e3]
    if not chs:
        return None, f"no signal at {cf/1e6:.1f} MHz to tune on"
    sig = max(chs, key=lambda c: c["peak"])["freq"]*1e6
    fr, P = gain_search._spectrum(x, FS, cf); pw = 10**(P/10)
    best = None
    for c0 in np.arange(cf - 3.4e6, cf + 3.0e6, 100e3): # quietest 400 kHz away from the DC spike, the signal and the other lines
        if abs(c0 + 200e3 - cf) < 400e3 or abs(c0 + 200e3 - sig) < 500e3 or any(abs(c0 + 200e3 - c["freq"]*1e6) < 400e3 for c in chs):
            continue
        m = np.mean(pw[(fr >= c0) & (fr < c0 + 400e3)])
        if best is None or m < best[0]:
            best = (m, c0)
    if best is None:
        return None, "no empty band"
    gain, rows = gain_search.search(sdr, cf, FS, (sig - 15e3, sig + 15e3), (best[1], best[1] + 400e3), verbose=False)
    r = max(rows, key=lambda r: r.score)
    return gain, f"signal {sig/1e6:.4f} MHz, empty {best[1]/1e6:.2f}-{(best[1] + 400e3)/1e6:.2f} MHz: SNR {r.snr:.1f} dB, clip {r.clip:.0e}, {len(r.artifacts)} artifact lines"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bands", help="MHz ranges, e.g. 118-174,400-470 (default: " + ",".join(f"{l}-{h}" for l, h in BANDS) + ")")
    ap.add_argument("--seconds", type=float, default=8.0); ap.add_argument("--max", type=int, default=6, help="classify at most this many channels per window")
    ap.add_argument("--min-peak", type=float, default=10.0, help="channels must reach this many dB over the window median"); ap.add_argument("--wav-dir")
    ap.add_argument("--no-tune", action="store_true", help="skip the per-band gain search: highest gain that does not clip, per window")
    a = ap.parse_args(argv)
    bands = [tuple(float(v) for v in b.split("-")) for b in a.bands.split(",")] if a.bands else BANDS
    if a.wav_dir:
        os.makedirs(a.wav_dir, exist_ok=True)
    sdr = radio_io.open_radio(cf=100e6, fs=FS); radio_io.start_rx(sdr)
    results = []
    try:
        for blo, bhi in bands:
            gain = None
            if not a.no_tune:
                gain, why = tune_band(sdr, blo, bhi)
                print(f"\nband {blo:g}-{bhi:g} MHz: gain {f'{gain:.2f}' if gain is not None else 'by ladder'} ({why})", flush=True)
            else:
                print(f"\nband {blo:g}-{bhi:g} MHz", flush=True)
            cf = blo*1e6 + FS/2
            while cf - FS/2 < bhi*1e6:
                sdr.set_rx_freq(cf, flush=True)
                if gain is not None:
                    radio_io.set_gain(sdr, gain); time.sleep(0.05)
                    x = radio_io.grab(sdr, int(FS*a.seconds)); clip = radio_io.clipped(x); g = gain
                    if clip > 1e-4: # a strong signal in this window: back off from the tuned setting
                        x, g, clip = radio_io.capture_clean(sdr, FS, a.seconds, check_seconds=0.5)
                else:
                    x, g, clip = radio_io.capture_clean(sdr, FS, a.seconds, check_seconds=0.5)
                x = x.astype(np.complex64)
                lo = max(blo*1e6, cf - STEP/2); hi = min(bhi*1e6, cf + STEP/2)
                chs, _, _ = channel_watch.channels(x, FS, cf, lo, hi, 12.5e3, 0.02, 8.0, 0.01, a.min_peak)
                chs = sorted([c for c in chs if not c["spur"]], key=lambda c: -c["peak"])[:a.max]
                print(f"window {cf/1e6:7.1f} MHz: gain {g:.2f}, clipped {clip:.0e}, {len(chs)} channels", flush=True)
                for c in chs:
                    lab, ft, aud = classify(x, FS, cf, c["freq"]*1e6)
                    line = f"   {c['freq']:9.4f} MHz  duty {100*c['duty']:5.1f}%  +{c['peak']:4.1f} dB  {lab}"
                    if ft:
                        line += f"   (dev {ft.get('dev', 0)/1e3:.1f} kHz, voice band {ft['voice']:.0f}%, syllabic {ft['syl']:.1f})"
                    if aud is not None and a.wav_dir:
                        path = os.path.join(a.wav_dir, f"{c['freq']:.4f}_{lab.split()[0].lower()}.wav"); write_wav(path, *aud); line += f"  -> {path}"
                    print(line, flush=True); results.append((c["freq"], lab, c["duty"], c["peak"]))
                del x
                cf += STEP
    finally:
        sdr.stop_rx(); sdr.close()
    print("\nsummary:")
    for lab in ("FM voice", "AM voice", "P25", "DMR", "other", "carrier"):
        r = sorted(r_ for r_ in results if r_[1] == lab)
        if r:
            print(f"  {lab}: " + ", ".join(f"{f:.4f}" for f, _, _, _ in r))
    return results


if __name__ == "__main__":
    main()
