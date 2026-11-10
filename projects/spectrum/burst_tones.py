"""Bursts of one narrowband channel: count, timing, FM statistics, and the tone in each burst

    python projects/spectrum/burst_tones.py data/fed/fed_142.4.cf64 --fs 2 --cf 142.4 --freq 142.8525
    python projects/spectrum/burst_tones.py CAPTURE --fs 2 --cf 142.4 --freq 142.8525 --list 100 --wav bursts.wav

The channel at --freq is mixed to zero, low-passed and decimated to 50 kS/s; bursts are runs of 2 ms frames more than --thresh dB over the channel's median power (at least --min-ms long).
Reported: burst count and durations, envelope cv (constant-envelope FM is ~0.05-0.2, voice AM > 0.3), carrier offset from --freq (subtract the ~15 ppm HackRF clock error for the true
channel), frequency deviation, 99 % power bandwidth, a histogram of the instantaneous frequency (two peaks = 2-FSK, one bell = analog FM or noise), and for each burst the dominant FM-audio
frequency with its peak/median ratio (short bursts give unreliable values; a tone that repeats across many bursts is real). `--list N` prints the first N bursts in time order with the gap to
the previous one. `--wav` writes the demodulated bursts (silence between) for listening. Meant to be upstreamed into RPPS.
"""
import argparse
import wave

import numpy as np
from scipy.signal import resample_poly

DEC = 40


def channel(x, fs, cf, freq):
    """Complex baseband of the channel at `freq` (Hz), fs/DEC samples per second"""
    t = np.arange(len(x))
    y = x*np.exp(-2j*np.pi*(freq - cf)*t/fs).astype(np.complex64)
    return resample_poly(y, 1, DEC, window=("kaiser", 8.0)), fs/DEC


def find_bursts(z, fs, thresh=10.0, min_ms=6.0, frame_s=0.002):
    """(starts, ends) in frames of frame_s, and the frame power in dB"""
    fr = int(frame_s*fs); n = len(z)//fr
    p = 10*np.log10((np.abs(z[:n*fr]).reshape(n, fr)**2).mean(1) + 1e-18)
    on = p > np.median(p) + thresh
    idx = np.flatnonzero(np.diff(np.r_[0, on.astype(int), 0]))
    st, en = idx[::2], idx[1::2]
    k = (en - st)*frame_s*1e3 >= min_ms
    return st[k], en[k], p, fr


def fm(s, fs):
    return np.diff(np.unwrap(np.angle(s)))*fs/(2*np.pi)


def tone(f_, fs, lo=200.0, hi=8000.0):
    """Dominant frequency of an FM-audio burst and its peak/median ratio"""
    X = np.abs(np.fft.rfft(f_*np.hanning(len(f_)), 2048)); fq = np.fft.rfftfreq(2048, 1/fs)
    s = (fq > lo) & (fq < hi); k = np.argmax(X[s])
    return fq[s][k], X[s][k]/np.median(X[s])


def analyse(z, fs, thresh=10.0, min_ms=6.0):
    st, en, p, fr = find_bursts(z, fs, thresh, min_ms)
    pad = int(0.004*fs); rows = []; devs = []; cvs = []; cen = []; bws = []; fis = []
    for a, b in zip(st, en):
        s = z[max(0, a*fr - pad):b*fr + pad]
        if len(s) < 2*pad + 100:
            continue
        f_ = fm(s, fs); f_ = f_ - np.median(f_); core = f_[pad:-pad]; sc = s[pad:-pad]
        env = np.abs(sc); cvs.append(env.std()/env.mean())
        fi = core[env[1:] > 0.5*np.median(env)] if len(env) > 1 else core
        cen.append(np.median(fm(sc, fs))); devs.append(np.std(fi)); fis.append(fi)
        S = np.abs(np.fft.fftshift(np.fft.fft(sc*np.hanning(len(sc)), 4096)))**2; ff = np.fft.fftshift(np.fft.fftfreq(4096, 1/fs)); c = np.cumsum(S)/S.sum()
        bws.append(ff[np.searchsorted(c, 0.995)] - ff[np.searchsorted(c, 0.005)])
        tf, q = tone(core, fs)
        rows.append(dict(start=a*fr/fs, ms=len(core)/fs*1e3, tone=tf, q=q, peak=p[a:b].max() - np.median(p)))
    return rows, dict(cv=np.median(cvs), centre=np.median(cen), dev=np.median(devs), bw=np.median(bws), fi=np.concatenate(fis) if fis else np.array([]))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file"); ap.add_argument("--fs", type=float, required=True, help="[MS/s]"); ap.add_argument("--cf", type=float, required=True, help="capture centre [MHz]")
    ap.add_argument("--freq", type=float, required=True, help="channel [MHz]")
    ap.add_argument("--thresh", type=float, default=10.0); ap.add_argument("--min-ms", type=float, default=6.0)
    ap.add_argument("--list", type=int, default=0, help="print the first N bursts"); ap.add_argument("--wav", help="write the demodulated bursts here")
    a = ap.parse_args(argv)
    fs0 = a.fs*1e6; x = np.fromfile(a.file, np.complex64)
    z, fs = channel(x, fs0, a.cf*1e6, a.freq*1e6)
    rows, st = analyse(z, fs, a.thresh, a.min_ms)
    if not rows:
        print("no bursts"); return rows
    ms = np.array([r["ms"] for r in rows])
    print(f"{len(rows)} bursts >= {a.min_ms:g} ms in {len(x)/fs0:.0f} s; duration median {np.median(ms):.0f} ms, range {ms.min():.0f}-{ms.max():.0f}; median peak {np.median([r['peak'] for r in rows]):.1f} dB over the channel floor")
    print(f"envelope cv {st['cv']:.2f}; carrier {a.freq + st['centre']/1e6:.5f} MHz (measured; true ~ /1.000015 for the HackRF clock: {(a.freq + st['centre']/1e6)/1.000015:.5f}); deviation sd {st['dev']/1e3:.2f} kHz; 99% bw {st['bw']/1e3:.1f} kHz")
    h, _ = np.histogram(st["fi"], bins=np.arange(-8e3, 8.1e3, 1e3)); print("instantaneous frequency histogram, 1 kHz bins -8..8 kHz:", h.tolist())
    tonal = [r for r in rows if r["q"] > 8]
    print(f"tonal bursts (peak/median > 8): {len(tonal)} of {len(rows)}")
    if tonal:
        t = np.array([r["tone"] for r in tonal]); hh, e = np.histogram(t, bins=np.arange(200, 8001, 100))
        print("most common tones (Hz, count):", [(int(e[j]), int(hh[j])) for j in np.argsort(hh)[-8:][::-1] if hh[j]])
    prev = None
    if a.list:
        print("start s | ms | gap ms | tone Hz (peak/median)")
    for r in rows[:a.list]:
        gap = 0 if prev is None else (r["start"] - prev)*1e3
        print(f"{r['start']:8.3f} {r['ms']:4.0f} {gap:7.0f} {r['tone']:6.0f} ({r['q']:.0f})"); prev = r["start"] + r["ms"]/1e3
    if a.wav:
        st_, en_, p, fr = find_bursts(z, fs, a.thresh, a.min_ms); out = []
        for s0, e0 in zip(st_, en_):
            f_ = fm(z[s0*fr:e0*fr], fs); out += [f_ - np.median(f_), np.zeros(int(0.1*fs))]
        y = np.concatenate(out); y = (y/np.abs(y).max()*30000).astype(np.int16)
        with wave.open(a.wav, "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(int(fs)); w.writeframes(y.tobytes())
        print(f"wrote {a.wav} ({len(y)/fs:.1f} s at {fs:g} Hz)")
    return rows


if __name__ == "__main__":
    main()
