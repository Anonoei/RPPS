"""5G NR cell search on a capture: SSB positions, subcarrier spacing, N_ID2 by the PSS, N_ID1 by the SSS, the cell ID

    python projects/nr/nr_cell.py capture.cs16 --format cs16 --fs 100 --centres -30,0,30

NR (3GPP TS 38.211 7.4.2) sends its synchronization signals in SS/PBCH blocks (SSB): 240 subcarriers, 4 OFDM symbols,
repeated every 5-160 ms (20 ms is the default a UE assumes). The PSS (symbol 0) and SSS (symbol 2) take the 127 central
subcarriers. Unlike LTE's Zadoff-Chu PSS they are BPSK m-sequences:
  PSS d(n) = 1 - 2 x((n + 43 N_ID2) mod 127),  x(i+7) = x(i+4) + x(i),  x(0..6) = 0 1 1 0 1 1 1
  SSS d(n) = [1 - 2 x0((n + m0) mod 127)] [1 - 2 x1((n + m1) mod 127)],  m0 = 15 floor(N_ID1/112) + 5 N_ID2,
          m1 = N_ID1 mod 112,  x0(i+7) = x0(i+4) + x0(i),  x1(i+7) = x1(i+1) + x1(i),  both from 1 0 0 0 0 0 0
The SSB sits on the sync raster, not at the carrier's centre, so the search covers the band:
  1. PSS: an STFT at the subcarrier spacing, each frame's spectrum correlated with the PSS at every subcarrier shift.
     Differentially (X[k] X*[k-1] against d(n) d(n-1)): a window a few samples off the symbol start puts a phase ramp
     across the subcarriers that cancels a coherent sum, but not this one. 1.0 is a perfect match; noise and the other
     N_ID2 (the same m-sequence, cyclically shifted) stay well under the real one's
  2. fine: frequency and timing by time-domain correlation, summed over every occurrence at the period found
  3. SSS: equalized by the PSS (the channel on the same subcarriers), correlated with all 336 N_ID1
Sample rate: anything; each centre is resampled to 30.72 MS/s (a 30 MHz span) around it.
"""
import argparse
from math import gcd

import numpy as np
from scipy.signal import resample_poly

FS = 30.72e6
CP = {15e3: 144, 30e3: 72} # Normal prefix at 30.72 MS/s, past the first symbol of each half subframe

def _mseq(feedback, init):
    x = list(init)
    for i in range(127 - 7):
        x.append(feedback(x, i))
    return np.array(x)

X_PSS = _mseq(lambda x, i: (x[i + 4] + x[i]) % 2, [0, 1, 1, 0, 1, 1, 1])
X0 = _mseq(lambda x, i: (x[i + 4] + x[i]) % 2, [1, 0, 0, 0, 0, 0, 0])
X1 = _mseq(lambda x, i: (x[i + 1] + x[i]) % 2, [1, 0, 0, 0, 0, 0, 0])

def pss(nid2):
    return 1 - 2*X_PSS[(np.arange(127) + 43*nid2) % 127]

def sss(nid1, nid2):
    n = np.arange(127)
    m0 = 15*(nid1//112) + 5*nid2
    m1 = nid1 % 112
    return (1 - 2*X0[(n + m0) % 127])*(1 - 2*X1[(n + m1) % 127])

def to_rate(x, fs, centre):
    """x shifted so `centre` (Hz) is at 0 and resampled to FS"""
    n = np.arange(len(x))
    y = x*np.exp(-2j*np.pi*centre/fs*n).astype(np.complex64)
    up, down = int(FS/1e3), int(fs/1e3)
    g = gcd(up, down)
    return resample_poly(y, up//g, down//g)

def pss_search(y, scs, hop_div=32, chunk=2000, top=40):
    """[(metric, time s, offset Hz of the PSS centre, N_ID2)] best first: the differential STFT search"""
    N = int(FS/scs)
    hop = N//hop_div
    nf = (len(y) - N)//hop
    idx = np.arange(N)[None, :] + hop*np.arange(nf)[:, None]
    hits = []
    for s0 in range(0, nf, chunk):
        F = np.fft.fft(y[idx[s0:s0 + chunk]], axis=1)
        G = F[:, 1:]*np.conj(F[:, :-1])
        G = np.concatenate((G, np.zeros((len(G), 1))), axis=1)
        A = np.abs(G)**2
        FA = np.fft.fft(A, axis=1)
        FG = np.fft.fft(G, axis=1)
        for nid2 in range(3):
            p = pss(nid2)
            d = np.zeros(N)
            d[:126] = p[1:]*p[:-1]
            C = np.fft.ifft(FG*np.conj(np.fft.fft(d))[None, :], axis=1)
            E = np.fft.ifft(FA*np.conj(np.fft.fft((d != 0).astype(float)))[None, :], axis=1).real
            M = np.abs(C)**2/(126*np.maximum(E, 1e-12))
            for a, b in zip(*np.unravel_index(np.argsort(M, axis=None)[-top:], M.shape)):
                k = (b + 63) % N
                hits.append((float(M[a, b]), (s0 + a)*hop/FS, (k - N if k > N//2 else k)*scs, nid2))
    hits.sort(reverse=True)
    return hits

def cells(hits, min_repeats=2, tol_t=0.5e-3, tol_f=None):
    """Hits grouped by (offset, N_ID2): [(mean metric, offset, nid2, sorted times)], strongest first"""
    groups = []
    for m, t, f, n in hits:
        for g in groups:
            if g[2] == n and abs(g[1] - f) <= (tol_f or 20e3):
                if all(abs(t - u) > tol_t for u in g[3]):
                    g[3].append(t)
                    g[0].append(m)
                break
        else:
            groups.append([[m], f, n, [t]])
    out = [(float(np.mean(ms)), f, n, sorted(ts)) for ms, f, n, ts in groups if len(ts) >= min_repeats]
    return sorted(out, key=lambda g: -g[0])

def refine(y, scs, nid2, offset, times, search_hz=None, search_samples=None):
    """(metric, offset Hz, sample index of each PSS start): time-domain PSS correlation summed over the occurrences"""
    N = int(FS/scs)
    k = np.arange(-63, 64) % N
    search_hz = search_hz or scs
    span = search_samples or N + 200
    starts = [int(t*FS) for t in times]
    best = (0.0, offset, starts)
    for df in np.arange(-search_hz, search_hz + 1, scs/30):
        X = np.zeros(N, complex)
        X[k] = pss(nid2)
        tpl = np.fft.ifft(X)*np.exp(2j*np.pi*(offset + df)*np.arange(N)/FS)
        T = np.conj(tpl)[::-1]
        acc, pos = 0.0, []
        for s in starts:
            lo = max(0, s - span)
            seg = y[lo:s + N + span]
            if len(seg) < N + 1:
                continue
            r = np.abs(np.convolve(seg, T, mode="valid"))**2/(np.sum(np.abs(T)**2)*np.convolve(np.abs(seg)**2, np.ones(N), mode="valid"))
            i = int(np.argmax(r))
            acc += r[i]
            pos.append(lo + i)
        m = acc/max(len(pos), 1)
        if m > best[0]:
            best = (m, offset + df, pos)
    return best

def sss_decode(y, scs, nid2, offset, starts):
    """(N_ID1, score, second, median): the SSS two symbols after each PSS, equalized by it, against all 336"""
    N, cp = int(FS/scs), CP[scs]
    k = np.arange(-63, 64) % N
    votes = np.zeros(336)
    S = np.array([sss(i, nid2) for i in range(336)])
    used = 0
    for s in starts:
        seg = y[s:s + 3*(N + cp) + N]
        if len(seg) < 3*(N + cp):
            continue
        seg = seg*np.exp(-2j*np.pi*offset*np.arange(s, s + len(seg))/FS)
        H = np.fft.fft(seg[:N])[k]*pss(nid2)
        st = 2*(N + cp)
        Z = np.fft.fft(seg[st:st + N])[k]*np.conj(H)
        votes += np.abs(S @ Z)/np.sum(np.abs(Z))
        used += 1
    if not used:
        return None, 0.0, 0.0, 0.0
    v = votes/used
    o = np.argsort(v)[::-1]
    return int(o[0]), float(v[o[0]]), float(v[o[1]]), float(np.median(v))

def burst_period(times):
    """In words: SSBs come in bursts (one per beam, a fraction of a ms apart) repeating every 5-160 ms"""
    d = np.diff(times)*1e3
    if not len(d):
        return "once"
    near, far = d[d < 4.0], d[d >= 4.0]
    out = []
    if len(far):
        out.append(f"bursts every {np.median(far):.1f} ms")
    if len(near):
        out.append(f"{len(near)//max(len(far) + 1, 1) + 1} a burst, {np.median(near):.2f} ms apart")
    return ", ".join(out)

def load(path, fmt, seconds=None, fs=None):
    if fmt == "cs16":
        x = np.fromfile(path, np.int16, count=-1 if seconds is None else int(2*seconds*fs))
        return x.astype(np.float32).view(np.complex64)
    return np.fromfile(path, np.complex64, count=-1 if seconds is None else int(seconds*fs))

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path")
    ap.add_argument("--format", choices=("cs16", "cf32"), default="cf32", help="Interleaved int16 or complex64 samples")
    ap.add_argument("--fs", type=float, required=True, help="Sample rate [MS/s]")
    ap.add_argument("--cf", type=float, default=0.0, help="Capture centre [MHz], to print absolute frequencies")
    ap.add_argument("--centres", default="0", help="Comma-separated centres to search around [MHz from the capture centre]")
    ap.add_argument("--scs", default="15,30", help="Subcarrier spacings to try [kHz]")
    ap.add_argument("--seconds", type=float, default=None, help="Read only this much")
    ap.add_argument("--min-metric", type=float, default=0.3, help="A PSS group's mean metric to decode (noise stays ~0.15)")
    a = ap.parse_args(argv)
    fs = a.fs*1e6
    x = load(a.path, a.format, a.seconds, fs)
    print(f"{a.path}: {len(x)/fs:.3f} s at {a.fs} MS/s")
    found = 0
    for c in (float(v)*1e6 for v in a.centres.split(",")):
        y = to_rate(x, fs, c)
        for scs in (float(v)*1e3 for v in a.scs.split(",")):
            groups = cells(pss_search(y, scs))
            print(f"\ncentre {c/1e6:+.2f} MHz, SCS {scs/1e3:.0f} kHz: "
                  + (", ".join(f"N_ID2 {n} at {(c + f)/1e6:+.3f} MHz x{len(ts)} ({m:.2f})" for m, f, n, ts in groups[:4]) or "no repeating PSS"))
            for m, f, n, ts in groups:
                if m < a.min_metric:
                    break
                per = burst_period(ts)
                fm, off, starts = refine(y, scs, n, f, ts)
                nid1, s1, s2, med = sss_decode(y, scs, n, off, starts)
                ok = nid1 is not None and s1 > 2*s2 - med and s1 > 0.5
                found += ok
                print(f"  SSB at {(c + off)/1e6:+.4f} MHz" + (f" ({a.cf + (c + off)/1e6:.4f} MHz)" if a.cf else "") +
                      f": N_ID2 {n}, {per}, {len(starts)} seen, PSS {fm:.2f}; SSS N_ID1 {nid1} "
                      f"({s1:.2f}, next {s2:.2f}, median {med:.2f}) => " + (f"PCI {3*nid1 + n}" if ok else "no clear SSS"))
    return 0 if found else 1

if __name__ == "__main__":
    raise SystemExit(main())
