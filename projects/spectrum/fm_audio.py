"""Narrowband FM audio from one channel of a capture, with a check for whether it is voice, a sub-audible tone (CTCSS) or just a carrier

    python projects/spectrum/fm_audio.py data/lmr/lmr_462.5.cf64 --fs 8 --cf 462.5 --freq 464.8062 --wav data/lmr/464.8062.wav

The channel is mixed to zero, low-passed (--bw kHz), FM demodulated, de-emphasised (75 us) and decimated to 16 kS/s. Reported: the frequency deviation (rms and p99), the share of audio energy in
sub-audible (<300 Hz), voice (300-3000 Hz) and high (>3 kHz) bands, the strongest tone below 260 Hz (CTCSS: 67-254 Hz), and the syllabic rate: speech has its envelope in the 300-3000 Hz band modulated at
2-8 Hz, a carrier with a tone or hiss does not. `--wav` writes the audio, silent where the carrier is absent. Nothing is decoded and nothing is uploaded. Meant to be upstreamed into RPPS.
"""
import argparse
import wave

import numpy as np
from rpps.signal.filters import IIR, iir, low_pass
from rpps.signal.sample import resample_poly

import burst_tones

CTCSS = [67.0, 69.3, 71.9, 74.4, 77.0, 79.7, 82.5, 85.4, 88.5, 91.5, 94.8, 97.4, 100.0, 103.5, 107.2, 110.9, 114.8, 118.8, 123.0, 127.3, 131.8, 136.5, 141.3, 146.2, 151.4, 156.7, 162.2, 167.9, 173.8,
         179.9, 186.2, 192.8, 203.5, 210.7, 218.1, 225.7, 233.6, 241.8, 250.3]


def baseband(x, fs, cf, freq, bw=12e3):
    """(complex baseband of the channel, its rate [Hz]): mixed to the strongest line near freq, low-passed to bw"""
    z, fs1 = burst_tones.channel(x, fs, cf, freq)
    n = 2**14; k = len(z)//n
    S = (np.abs(np.fft.fftshift(np.fft.fft(z[:k*n].reshape(k, n)*np.hanning(n), axis=1), axes=1))**2).mean(0); ff = np.fft.fftshift(np.fft.fftfreq(n, 1/fs1))
    c = ff[np.argmax(np.convolve(S, np.ones(8), "same"))]
    z = z*np.exp(-2j*np.pi*c*np.arange(len(z))/fs1)
    return low_pass(201, bw, fs=fs1).run(z), fs1


def carrier_mask(z, fs1, n_out):
    """Where the carrier is present (within 8 dB of its 95th-percentile power in 10 ms frames), at 16 kS/s"""
    p = np.abs(z)**2; fr = int(fs1/100); e = 10*np.log10(p[:len(p)//fr*fr].reshape(-1, fr).mean(1) + 1e-12)
    on = e > np.percentile(e, 95) - 8
    mask = np.repeat(on, 160)[:n_out]
    return np.pad(mask, (0, n_out - len(mask)))


def fm_demod(z, fs1):
    """(audio at 16 kS/s with 75 us de-emphasis, carrier mask, deviation samples [Hz])"""
    dev = np.diff(np.unwrap(np.angle(z)))*fs1/(2*np.pi)
    a = resample_poly(dev, 1, int(round(fs1/16000)))
    a = IIR(iir.design("butter", 1, "lowpass", 1/(2*np.pi*75e-6)/16000)).run(a)
    return a, carrier_mask(z, fs1, len(a)), dev


def am_demod(z, fs1):
    """(envelope audio at 16 kS/s, carrier mask): the envelope with its mean removed per second, so slow fading does not count"""
    e = np.abs(z); a = resample_poly(e, 1, int(round(fs1/16000)))
    return a, carrier_mask(z, fs1, len(a))


def features(audio, mask):
    """Is this audio speech? dict(frac: carrier present, voice: % of audio energy at 300-3000 Hz, low: % below 300, tone Hz and dB over the sub-audible band median, syl: syllabic ratio)"""
    au = audio[mask]
    if len(au) < 16000:
        return None
    au = au - np.mean(au); au = au[:2**18]
    A = np.abs(np.fft.rfft(au*np.hanning(len(au))))**2; fq = np.fft.rfftfreq(len(au), 1/16000); tot = A[fq > 20].sum()
    band = lambda lo, hi: 100*A[(fq >= lo) & (fq < hi)].sum()/tot
    low = (fq > 60) & (fq < 260); k = np.argmax(A[low])
    v = IIR(iir.design("butter", 4, "bandpass", [300/16000, 3000/16000])).run(au); env = resample_poly(np.abs(v), 1, 160)
    E = np.abs(np.fft.rfft((env - env.mean())*np.hanning(len(env))))**2; fe = np.fft.rfftfreq(len(env), 1/100)
    syl = E[(fe >= 2) & (fe <= 8)].sum()/max(1e-12, E[(fe >= 8) & (fe <= 30)].sum())
    return dict(frac=mask.mean(), voice=band(300, 3000), low=band(20, 300), high=band(3000, 8000), tone=fq[low][k], tone_db=10*np.log10(A[low].max()/np.median(A[low])), syl=syl)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file"); ap.add_argument("--fs", type=float, required=True, help="[MS/s]"); ap.add_argument("--cf", type=float, required=True, help="[MHz]")
    ap.add_argument("--freq", type=float, required=True); ap.add_argument("--bw", type=float, default=12.0, help="channel low-pass [kHz]"); ap.add_argument("--wav")
    a = ap.parse_args(argv)
    x = np.fromfile(a.file, np.complex64)
    z, fs1 = baseband(x, a.fs*1e6, a.cf*1e6, a.freq*1e6, a.bw*1e3)
    audio, mask, dev = fm_demod(z, fs1)
    ft = features(audio, mask)
    if ft is None:
        print("carrier present for less than 1 s"); return
    cts = min(CTCSS, key=lambda t: abs(t - ft["tone"]))
    print(f"{a.freq:.4f} MHz: carrier present {100*ft['frac']:.0f}% of the time; deviation rms {np.std(dev)/1e3:.2f} kHz")
    print(f"  audio energy: <300 Hz {ft['low']:.0f}%, 300-3000 Hz {ft['voice']:.0f}%, >3 kHz {ft['high']:.0f}%; strongest tone below 260 Hz {ft['tone']:.1f} Hz (nearest CTCSS {cts} Hz), {ft['tone_db']:.0f} dB over the band median")
    print(f"  syllabic ratio (2-8 Hz / 8-30 Hz envelope modulation) {ft['syl']:.1f}   (speech is typically > 2; a steady tone or noise is < 1)")
    if a.wav:
        y = np.where(mask, audio, 0.0); y = y/np.abs(y).max()*0.8
        with wave.open(a.wav, "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000); w.writeframes((y*32767).astype(np.int16).tobytes())
        print(f"  wrote {a.wav} ({len(y)/16000:.1f} s)")


if __name__ == "__main__":
    main()
