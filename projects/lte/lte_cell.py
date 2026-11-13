"""LTE cell search on a capture: carrier offset, cell ID, frame timing

    python projects/lte/lte_cell.py data/lte_b13.cf64 --fs 15.36 --cf 753 --carrier 750.9

LTE downlink (3GPP TS 36.211, FDD, normal cyclic prefix). Every 5 ms the central 62 subcarriers of the last OFDM symbol of subframes 0
and 5 carry the primary synchronization signal (PSS: one of three Zadoff-Chu roots, which names N_ID2 = 0..2), and the symbol before
it carries the secondary (SSS: two interleaved m-sequences that name N_ID1 = 0..167 and which half-frame this is). The cell ID is
3*N_ID1 + N_ID2. Steps, each with a statistic that is near chance when it is wrong:
  1. PSS: correlate at 1.92 MS/s with the three roots over a grid of frequency offsets (a HackRF is ~15 ppm off, which at 750 MHz is nearly a
     whole 15 kHz subcarrier, so the offset has to be searched); fold on 5 ms; the right (root, offset) has a sharp peak, others a flat floor
  2. cyclic prefix: the first 7% of every OFDM symbol repeats its last 7%: the phase of that correlation is the fractional-subcarrier offset
  3. SSS: equalize with the PSS and correlate with all 168 x 2 hypotheses
Sample rates: 1.92 MS/s per 1.4 MHz of bandwidth: 1.92 (FFT 128), 3.84 (256), 7.68 (512), 15.36 (1024), 23.04 (1536), 30.72 (2048).
"""
import argparse
import sys

import numpy as np

SCS = 15e3
ROOTS = (25, 29, 34) # N_ID2 = 0, 1, 2

def pss_sequence(u):
    n = np.arange(62)
    return np.where(n < 31, np.exp(-1j*np.pi*u*n*(n + 1)/63), np.exp(-1j*np.pi*u*(n + 1)*(n + 2)/63))

def bins62(N):
    """FFT bin of each of the 62 central subcarriers (k = -31..-1, +1..+31; the DC subcarrier is not used)"""
    k = np.concatenate((np.arange(-31, 0), np.arange(1, 32)))
    return k % N

def pss_time(u, N):
    X = np.zeros(N, dtype=complex)
    X[bins62(N)] = pss_sequence(u)
    return np.fft.ifft(X)

def _mseq(taps):
    x = [0, 0, 0, 0, 1]
    for i in range(26):
        x.append(sum(x[i + t] for t in taps) % 2)
    return 1 - 2*np.array(x)

S_TILDE = _mseq((2, 0)) # x(i+5) = x(i+2) + x(i)
C_TILDE = _mseq((3, 0)) # x(i+5) = x(i+3) + x(i)
Z_TILDE = _mseq((4, 2, 1, 0)) # x(i+5) = x(i+4) + x(i+2) + x(i+1) + x(i)

def sss_sequence(n_id1, n_id2, half):
    """The 62 SSS values for cell N_ID1, N_ID2 in subframe 0 (half = 0) or 5 (half = 1)"""
    qp = n_id1//30
    q = (n_id1 + qp*(qp + 1)//2)//30
    mp = n_id1 + q*(q + 1)//2
    m0 = mp % 31
    m1 = (m0 + mp//31 + 1) % 31
    n = np.arange(31)
    s0 = S_TILDE[(n + m0) % 31]; s1 = S_TILDE[(n + m1) % 31]
    c0 = C_TILDE[(n + n_id2) % 31]; c1 = C_TILDE[(n + n_id2 + 3) % 31]
    z0 = Z_TILDE[(n + (m0 % 8)) % 31]; z1 = Z_TILDE[(n + (m1 % 8)) % 31]
    d = np.empty(62)
    if half == 0:
        d[0::2] = s0*c0; d[1::2] = s1*c1*z0
    else:
        d[0::2] = s1*c0; d[1::2] = s0*c1*z1
    return d

def lowpass_decimate(x, ratio, cutoff=0.8e6, fs=15.36e6, taps=129):
    h = np.sinc(2*cutoff/fs*(np.arange(taps) - (taps - 1)/2))*np.hamming(taps)
    h /= h.sum()
    return np.convolve(x, h, "same")[::ratio]

def fft_correlate(x, template):
    """c[n] = sum_i x[n+i] conj(template[i])"""
    L = len(x) + len(template)
    nfft = 1 << (L - 1).bit_length()
    return np.fft.ifft(np.fft.fft(x, nfft)*np.conj(np.fft.fft(template, nfft)))[:len(x) - len(template) + 1]

def pss_search(x, fs, offsets, seconds=0.1):
    """For each frequency offset (Hz, where the carrier appears relative to the capture's centre) and each root: the fold of |PSS correlation|^2 on
    5 ms. Returns a list of (peak/median, offset, root index, fold) best first."""
    n = int(fs*seconds)
    t = np.arange(n)/fs
    ratio = int(round(fs/1.92e6))
    period = 9600 # 5 ms at 1.92 MS/s
    tmpl = [pss_time(u, 128) for u in ROOTS]
    out = []
    for off in offsets:
        y = lowpass_decimate(x[:n]*np.exp(-2j*np.pi*off*t), ratio, fs=fs)
        for r, tp in enumerate(tmpl):
            c = np.abs(fft_correlate(y, tp))**2
            m = len(c)//period*period
            fold = c[:m].reshape(-1, period).mean(0)
            out.append((fold.max()/np.median(fold), off, r, fold))
    out.sort(key=lambda v: -v[0])
    return out

def cp_metric(x, fs, start, nfft, n_symbols=70):
    """Cyclic-prefix correlation over `n_symbols` OFDM symbols from a slot start (the pattern is a 7-symbol slot, the first symbol with the
    longer prefix): a symbol's first Ncp samples repeat its last Ncp. Returns (coefficient, residual frequency offset [Hz]).
    The coefficient (|sum a b*| / sum |a||b|) is ~0.05 for noise and peaks only at the true symbol boundaries: it locates the grid.
    The phase of the sum is the offset within +-half a subcarrier: a later copy is advanced by 2*pi*df*N/fs."""
    scale = nfft/2048
    lens = [int(round(160*scale))] + [int(round(144*scale))]*6
    acc, en, pos = 0j, 0.0, start
    for s in range(n_symbols):
        c = lens[s % 7]
        if pos + c + nfft > len(x):
            break
        a, b = x[pos:pos + c], x[pos + nfft:pos + nfft + c]
        acc += np.sum(a*np.conj(b)); en += np.sum(np.abs(a)*np.abs(b))
        pos += c + nfft
    return abs(acc)/max(en, 1e-12), -np.angle(acc)/(2*np.pi*nfft/fs)

def align_slots(y, fs, nfft, approx, window=48):
    """Slot start (sample index) near `approx`, from the cyclic-prefix grid. The PSS peak is only good to a few samples, and a shift of one
    prefix length also scores well over a short window, so use many symbols (70)."""
    scores = [(cp_metric(y, fs, approx + sh, nfft)[0], approx + sh) for sh in range(-window, window + 1)]
    return max(scores)

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path")
    ap.add_argument("--fs", type=float, required=True, help="Sample rate [MS/s]")
    ap.add_argument("--cf", type=float, required=True, help="Capture centre [MHz]")
    ap.add_argument("--carrier", type=float, required=True, help="Where the carrier's centre should be [MHz]")
    ap.add_argument("--span", type=float, default=150.0, help="Search +-this [kHz] around it (default 150: the 100 kHz raster plus a clock error)")
    ap.add_argument("--step", type=float, default=5.0, help="Frequency step [kHz]")
    a = ap.parse_args(argv)
    fs, cf = a.fs*1e6, a.cf*1e6
    nfft = int(round(fs/SCS))
    x = np.fromfile(a.path, np.complex64).astype(np.complex128)
    print(f"{a.path}: {len(x)/fs:.2f} s at {fs/1e6:g} MS/s, centre {cf/1e6:g} MHz, FFT {nfft}")
    nominal = a.carrier*1e6 - cf
    offsets = nominal + np.arange(-a.span*1e3, a.span*1e3 + 1, a.step*1e3)
    t0 = None
    res = pss_search(x, fs, offsets)
    print("\nPSS search (peak / median of the 5 ms fold; chance is ~3-5):")
    best_per_root = {}
    for ratio, off, r, fold in res:
        best_per_root.setdefault(r, (ratio, off, fold))
    for r in range(3):
        ratio, off, _ = best_per_root[r]
        print(f"  root {ROOTS[r]} (N_ID2 = {r}): best offset {off/1e3:+8.1f} kHz from the capture centre, ratio {ratio:6.1f}")
    ratio, off, r, fold = res[0]
    print(f"  -> N_ID2 = {r}, carrier appears {off/1e3:+.1f} kHz from the centre = {(cf + off)/1e6:.4f} MHz "
          f"({(cf + off)/1e6 - a.carrier:+.4f} MHz from the nominal {a.carrier} MHz)")
    n_id2 = r
    # --- timing: occurrences of the PSS every 5 ms, from the fold's peak
    ratio_dec = int(round(fs/1.92e6))
    t = np.arange(len(x))/fs
    y = x*np.exp(-2j*np.pi*off*t)
    p0 = int(np.argmax(fold))*ratio_dec # sample (full rate) of the start of the PSS symbol's useful part, mod 5 ms
    period = int(round(0.005*fs))
    scale = nfft/2048
    cp = int(round(144*scale)); cp0 = int(round(160*scale))
    pss_off = (cp0 + nfft) + 5*(cp + nfft) + cp # useful start of symbol 6 of a slot, from the slot start: symbol 0 whole, symbols 1-5 whole, the prefix of 6
    # --- timing of each PSS occurrence, at the full sample rate. The HackRF's sample clock is ~15 ppm off, so occurrences drift by
    # ~1 sample per 5 ms and each one is located on its own, near where the previous predicted it
    tmpl = pss_time(ROOTS[n_id2], nfft)
    def refine(pos, half=24):
        seg = y[pos - half:pos + half + nfft]
        if pos - half < 0 or len(seg) < 2*half + nfft:
            return None
        c = np.abs(fft_correlate(seg, tmpl))
        j = int(np.argmax(c))
        return pos - half + j, c[j]/np.median(c)
    pss_pos = []
    pos = p0 - ratio_dec*0 # the fold's peak (decimated by ratio_dec: good to +-ratio_dec/2 samples)
    k = 0
    while True:
        guess = p0 + k*period if not pss_pos else pss_pos[-1] + period
        r_ = refine(guess, 30 if len(pss_pos) else 40)
        if r_ is None:
            break
        pss_pos.append(r_[0]); k += 1
    pss_pos = np.array(pss_pos)
    idx = np.arange(len(pss_pos))
    slope, icpt = np.polyfit(idx, pss_pos, 1)
    clock_ppm = (slope/period - 1)*1e6
    print(f"\nPSS timing: {len(pss_pos)} occurrences; spacing {slope:.2f} samples (nominal {period}) => sample clock {clock_ppm:+.1f} ppm "
          f"(HackRF + cell; scatter {np.std(pss_pos - (slope*idx + icpt)):.2f} samples)")
    # --- stage 2: fractional-subcarrier offset from the cyclic prefix, with the slot start taken from the PSS position
    dfs = [cp_metric(y, fs, int(pss_pos[i]) - pss_off, nfft, 20) for i in range(2, min(len(pss_pos) - 2, 40))]
    coef = np.mean([d[0] for d in dfs]); df = float(np.median([d[1] for d in dfs]))
    print(f"Cyclic prefix at those positions: coefficient {coef:.2f} (noise ~0.05); residual offset {df:+.0f} Hz "
          f"(spread {np.std([d[1] for d in dfs]):.0f}; must be within +-{SCS/2:.0f})")
    off_tot = off + df
    y = x*np.exp(-2j*np.pi*off_tot*t)
    print(f"  carrier at {(cf + off_tot)/1e6:.5f} MHz; nominal {a.carrier} MHz => {(cf + off_tot - a.carrier*1e6)/1e3:+.2f} kHz "
          f"({(cf + off_tot - a.carrier*1e6)/(a.carrier*1e6)*1e6:+.1f} ppm: the HackRF's clock and the cell's together)")
    # --- stage 3: SSS at each PSS occurrence
    seqs = {(n1, h): sss_sequence(n1, n_id2, h) for n1 in range(168) for h in (0, 1)}
    D = np.array([seqs[(n1, h)] for n1 in range(168) for h in (0, 1)]) # (336, 62)
    hyp = [(n1, h) for n1 in range(168) for h in (0, 1)]
    bins = bins62(nfft)
    d_pss = pss_sequence(ROOTS[n_id2])
    found = []
    for k in range(len(pss_pos)):
        t6 = int(pss_pos[k]) - 8 # a little early: inside the cyclic prefix
        t5 = t6 - (cp + nfft)
        if t5 < 0 or t6 + nfft > len(y):
            continue
        Y6 = np.fft.fft(y[t6:t6 + nfft])[bins]
        Y5 = np.fft.fft(y[t5:t5 + nfft])[bins]
        h = Y6/d_pss
        z = Y5*np.conj(h)/np.maximum(np.abs(h)**2, 1e-12)
        metric = np.abs(D @ z)/np.sqrt(np.sum(np.abs(z)**2)*62)
        order = np.argsort(-metric)
        found.append((k, hyp[order[0]], metric[order[0]], metric[order[1]], np.median(metric)))
    from collections import Counter
    votes = Counter(f[1][0] for f in found)
    n1, nv = votes.most_common(1)[0]
    print(f"\nSSS: {len(found)} occurrences; N_ID1 votes (top 3): {votes.most_common(3)}")
    good = [f for f in found if f[1][0] == n1]
    print(f"  best-vs-second metric on the agreeing ones: {np.mean([f[2] for f in good]):.2f} vs {np.mean([f[3] for f in good]):.2f} (chance level {np.mean([f[4] for f in good]):.2f})")
    halves = [f[1][1] for f in found if f[1][0] == n1]
    print(f"  half-frame (0 = subframe 0, 1 = subframe 5) by occurrence: {halves[:12]} ... (should alternate)")
    cell = 3*n1 + n_id2
    print(f"\nCell ID: N_ID1 = {n1}, N_ID2 = {n_id2}  =>  PCI {cell}")
    return dict(y=y, fs=fs, cf=cf, nfft=nfft, pss_pos=pss_pos, period=period, pci=cell, off=off_tot, found=found, n1=n1, n2=n_id2)

if __name__ == "__main__":
    main()
