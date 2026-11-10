"""Is this channel an FSK data signal, and at what baud rate? For finding EDACS (9600 baud 2-FSK control channels), MDC-1200 / POCSAG / FLEX-like paging, and other unidentified digital channels

    python projects/spectrum/fsk_probe.py data/lmr/p25_854.cf64 --fs 8 --cf 854 --freq 851.425 852.575 853.325

Per channel: the channel is mixed to zero and FM demodulated (fm_audio.baseband / fm_demod's discriminator, 15 kHz wide), restricted to the frames where the carrier is present. Reported:
  duty        share of 10 ms frames with the carrier present
  levels      the instantaneous-frequency distribution: two clear peaks = 2-level FSK (separation and spread are printed), four = 4-level (P25 C4FM / DMR 4FSK / NXDN), one bell = analog or noise
  baud        the strongest line, between 200 and 20000 Hz, in the spectrum of (frequency - its mean)^2 (an NRZ-like frequency signal has a line at the symbol rate once the filtering rounds the
              transitions); the table of well-known rates is searched: 9600 EDACS / ProVoice / GMSK, 4800 P25 / DMR / NXDN, 6000 P25 phase 2, 3600 / 2400 / 1200 / 512 paging and MDC
Not a decoder: it says whether a decoder for a given 2-FSK system is worth writing. Meant to be upstreamed into RPPS.
"""
import argparse

import numpy as np
from scipy.signal import resample_poly

import fm_audio

KNOWN = {9600: "EDACS / ProVoice / 9600 GFSK data", 4800: "P25 phase 1 / DMR / NXDN", 6000: "P25 phase 2", 3600: "EDACS-like / FLEX 3200-6400 family", 2400: "POCSAG 2400 / NXDN48",
         1200: "POCSAG 1200 / MDC-1200 / APRS", 512: "POCSAG 512", 3200: "FLEX 3200", 6400: "FLEX 6400", 19200: "wideband data"}


def probe(x, fs, cf, freq, bw=15e3):
    z, fs1 = fm_audio.baseband(x, fs, cf, freq, bw)
    dev = np.diff(np.unwrap(np.angle(z)))*fs1/(2*np.pi)
    p = np.abs(z)**2; fr = int(fs1/100); e = 10*np.log10(p[:len(p)//fr*fr].reshape(-1, fr).mean(1) + 1e-12)
    on = e > np.percentile(e, 95) - 8
    m = np.repeat(on, fr)[:len(dev)]; m = np.pad(m, (0, len(dev) - len(m)))
    d = dev[m]
    if len(d) < 20000:
        return None
    d = d - np.median(d)
    h, edges = np.histogram(d, bins=np.arange(-8e3, 8.01e3, 250)); c = (edges[:-1] + edges[1:])/2
    hs = np.convolve(h, np.ones(3)/3, "same")
    peaks = [i for i in range(1, len(hs) - 1) if hs[i] > hs[i - 1] and hs[i] >= hs[i + 1] and hs[i] > 0.25*hs.max()]
    # baud line: spectrum of d^2 (frequency signal's energy), searched over 200-20000 Hz at the demodulator rate fs1
    n = min(len(d), 2**20); q = d[:n]**2; q = q - q.mean()
    Q = np.abs(np.fft.rfft(q*np.hanning(n))); fq = np.fft.rfftfreq(n, 1/fs1)
    band = (fq > 200) & (fq < 20000); med = np.median(Q[band])
    top = np.argsort(Q[band])[-1]; f0 = fq[band][top]; rel = 20*np.log10(Q[band][top]/med)
    near = min(KNOWN, key=lambda b: abs(b - f0)/b)
    return dict(duty=on.mean(), peaks=[round(float(c[i])) for i in peaks], sd=float(np.std(d)), f0=float(f0), rel=float(rel), known=near if abs(near - f0)/near < 0.03 else None)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file"); ap.add_argument("--fs", type=float, required=True, help="[MS/s]"); ap.add_argument("--cf", type=float, required=True, help="[MHz]")
    ap.add_argument("--freq", type=float, nargs="+", required=True); ap.add_argument("--seconds", type=float, default=15.0); ap.add_argument("--bw", type=float, default=15.0, help="channel filter [kHz]")
    a = ap.parse_args(argv)
    fs = a.fs*1e6; x = np.fromfile(a.file, np.complex64, count=int(fs*a.seconds))
    out = {}
    for f in a.freq:
        r = probe(x, fs, a.cf*1e6, f*1e6, a.bw*1e3)
        if r is None:
            print(f"{f:9.4f} MHz: carrier present for under 0.2 s"); continue
        kind = {1: "one bell (analog / noise)", 2: "TWO levels (2-FSK)", 3: "three peaks", 4: "FOUR levels (4-FSK)"}.get(len(r["peaks"]), f"{len(r['peaks'])} peaks")
        print(f"{f:9.4f} MHz: duty {100*r['duty']:3.0f}%  levels: {kind} at {r['peaks']} Hz, sd {r['sd']/1e3:.2f} kHz  |  strongest baud-rate line {r['f0']:.0f} Hz (+{r['rel']:.0f} dB)"
              + (f" ~ {r['known']} baud: {KNOWN[r['known']]}" if r["known"] else ""))
        out[f] = r
    return out


if __name__ == "__main__":
    main()
