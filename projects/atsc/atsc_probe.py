"""Look at an ATSC (8VSB) digital TV channel in a capture: is it ATSC, how clean is it, where are its syncs

    python projects/atsc/atsc_probe.py data/atsc19.cf64 --fs 20 --cf 507 --lower 500 [--plot atsc.png]

--lower is the channel's lower edge [MHz] (US UHF channel n spans 470 + 6(n-14) to +6). What it does:
  1. finds the pilot (0.309 MHz above the lower edge) and measures the carrier offset
  2. centres the channel, filters it to 6 MHz, and locks to the pilot, leaving the real VSB signal
  3. resamples to 2 samples per symbol (10.762 MBd), RRC matched filter (roll-off 0.1152)
  4. finds the segment sync (+5 -5 -5 +5 every 832 symbols), which gives the symbol clock, and measures the clock error
  5. slices the data to the eight levels and reports the error vector (MER), unequalized
  6. finds the field syncs (a segment that repeats every 313 segments)
Only what the capture proves is reported. Decoding the transport stream needs an equalizer, the trellis decoder,
de-interleaving and Reed-Solomon: none of that is here.
"""
import argparse
import sys
from types import SimpleNamespace

import numpy as np

import rpps as rp
from rpps.signal.filters.shaping import rrc_pulse
from rpps.signal.sample.down import interpolate_stream

SYMBOL_RATE = 4.5e6*684/286 # 10.7622 MBd
PILOT_OFFSET = 309440.559 # Pilot above the channel's lower edge [Hz]
SEG = 832 # Symbols per data segment
FIELD = 313 # Segments per field
ROLLOFF = 0.1152

class ToneLock(rp.base.Stage):
    """Derotate a stream so a tone near `f0` Hz sits at DC (a phase-locked pilot)

    A block-rate PLL like fm_rx.PilotPLL: each `win` samples measure the tone's phase, then correct the
    oscillator. `st.err` collects the phase errors so the lock can be judged afterwards.
    """
    def __init__(self, f0, win=16384, kp=0.4, ki=0.05):
        self.f0, self.win, self.kp, self.ki = f0, win, kp, ki

    def new_state(self, meta):
        return SimpleNamespace(phase=0.0, freq=2*np.pi*self.f0/meta.Fs, err=[], level=[])

    def step(self, x, st, meta):
        out = np.empty_like(x)
        for i in range(0, len(x), self.win):
            c = x[i:i + self.win]
            n = len(c)
            ph = st.phase + st.freq*np.arange(n)
            d = c*np.exp(-1j*ph)
            out[i:i + n] = d
            if n >= self.win//4:
                z = np.mean(d)
                e = float(np.angle(z))
                st.err.append(e)
                st.level.append(abs(z))
                st.freq += self.ki*e/n
                st.phase += self.kp*e
            st.phase = (st.phase + st.freq*n) % (2*np.pi)
        return out

def find_pilot(x, Fs, cf, lower):
    """The pilot's frequency [Hz] near its nominal place, and its height over the data around it [dB]"""
    n = 2**17
    w = np.hanning(n); P = np.zeros(n); k = 0
    for i in range(0, min(len(x)//n, 32)*n, n): # 32 averaged spectra (0.2 s) place the pilot; the whole capture would only cost memory
        P += np.abs(np.fft.fft(x[i:i + n]*w))**2; k += 1
    P /= k
    fr = np.fft.fftfreq(n, 1/Fs) + cf
    nominal = lower + PILOT_OFFSET
    m = np.abs(fr - nominal) < 100e3
    i = np.where(m)[0][np.argmax(P[m])]
    y0, y1, y2 = np.log(P[i - 1:i + 2])
    delta = 0.5*(y0 - y2)/(y0 - 2*y1 + y2) # Parabolic peak
    flat = np.median(P[(np.abs(fr - nominal) > 30e3) & (np.abs(fr - nominal) < 300e3)])
    return fr[i] + delta*Fs/n, 10*np.log10(P[i]/flat)

def to_symbol_rate_signal(x, Fs, cf, f_pilot):
    """Centre, filter and pilot-lock. Returns the real VSB signal at Fs, and the pilot lock stage's state"""
    center = f_pilot + (3e6 - PILOT_OFFSET)
    pre = rp.signal.filters.ShiftFreq(-(center - cf)) + rp.signal.filters.low_pass(201, 3.0e6, Fs) # The 6 MHz channel, centred
    lock = ToneLock(-(3e6 - PILOT_OFFSET)) # The pilot is 2.69 MHz below the centre
    src = rp.source.File.__new__(rp.source.File) # unused: blocks come from the array below
    out = []
    state = None
    for blk in _blocks(x, Fs, cf, 2**20) + pre + lock:
        out.append(np.asarray(blk.data).real)
        state = blk.state(lock)
    return np.concatenate(out), state

def _blocks(x, Fs, cf, size):
    """Blocks of an array as a stream (the same Data a File source gives)"""
    from rpps.source.source import Blocks, Source
    class Arr(Source):
        def _samples(self, count):
            for i in range(0, len(x), count):
                yield self._block(x[i:i + count], flush=i + count >= len(x))
    return Arr(Fs, cf)(size)

def _convolve(a, h, mode="full"):
    """np.convolve, by FFT overlap-add when the filter is long enough to win"""
    out = rp.signal.filters.convolve(a, h, mode)
    return out.astype(a.dtype, copy=False) if np.isrealobj(a) and np.isrealobj(h) else out

def resample(y, step, pos0=1.0, block=1 << 20):
    """Cubic resampling at `step` input samples per output sample, in blocks (the same samples as one call, without gigabytes of temporaries for a long capture)"""
    st = SimpleNamespace(tail=None, pos=pos0)
    y = y.astype(np.float32, copy=False)
    return np.concatenate([interpolate_stream(y[i:i + block], st, step, "cubic") for i in range(0, len(y), block)])

def matched(y2):
    t = np.arange(-16, 17)/2 # +-8 symbols at 2 samples per symbol
    h = rrc_pulse(t, ROLLOFF)
    return _convolve(y2, h/np.sum(h), "same")

SYNC2 = np.array([5, 0, -5, 0, -5, 0, 5.0]) # Segment sync at 2 samples per symbol

def sync_fold(y, chunk=64):
    """Fold the segment-sync correlation on the 1664-sample segment. Returns the folded profile and the peak position
    in each `chunk` segments (fractional samples, unwrapped): a slope there is a clock error."""
    c = _convolve(y, SYNC2[::-1], "same")
    seg = 2*SEG
    n = len(c)//seg
    C = c[:n*seg].reshape(n, seg)
    prof = C.mean(0)
    pos = []
    for i in range(0, n - chunk + 1, chunk):
        f = C[i:i + chunk].mean(0)
        k = int(np.argmax(f))
        a, b, d = f[(k - 1) % seg], f[k], f[(k + 1) % seg]
        pos.append(k + 0.5*(a - d)/(a - 2*b + d))
    return prof, np.unwrap(np.array(pos), period=seg), seg

def slice_levels(d):
    """Nearest of the eight levels -7 -5 ... +7"""
    return np.clip(2*np.round((d - 1)/2) + 1, -7, 7)

MAX_TRAIN = 120000 # symbols an equalizer is trained on at most (97 unknowns: more rows only cost time; 0 uses every row)

def train_equalizer(y2, first, idx, target, L=96, chunk=200000, max_rows=None):
    """Least-squares equalizer: taps (L, at 2 samples per symbol) and a bias, so that the output at symbol n, i.e.
    sum_m taps[m]*y2[first + 2n + m - L//2] + bias, matches `target`. idx: the symbols n to train on.
    Normal equations are accumulated in chunks, so millions of symbols fit in memory."""
    sw = np.lib.stride_tricks.sliding_window_view(y2, L)
    A = np.zeros((L + 1, L + 1)); b = np.zeros(L + 1)
    idx = np.asarray(idx); target = np.asarray(target, dtype=np.float64)
    max_rows = MAX_TRAIN if max_rows is None else max_rows
    if max_rows and len(idx) > max_rows:
        keep = np.linspace(0, len(idx) - 1, max_rows).astype(int) # evenly spread over the block
        idx, target = idx[keep], target[keep]
    for i in range(0, len(idx), chunk):
        n = idx[i:i + chunk]
        pos = first + 2*n - L//2
        ok = (pos >= 0) & (pos + L <= len(y2))
        X = np.hstack([sw[pos[ok]], np.ones((int(ok.sum()), 1), dtype=np.float32)]).astype(np.float64)
        A += X.T @ X
        b += X.T @ target[i:i + chunk][ok]
    w = np.linalg.solve(A + 1e-9*np.trace(A)*np.eye(L + 1), b)
    return w[:L], w[L]

def apply_equalizer(y2, taps, bias, first, nsym):
    """The equalizer's output at each of `nsym` symbols"""
    L = len(taps)
    full = _convolve(y2, taps[::-1])
    idx = first + 2*np.arange(nsym) + (L - 1) - L//2
    return full[np.clip(idx, 0, len(full) - 1)] + bias

def equalize_range(y2, taps, bias, first, n0, n1):
    """The equalizer's output at symbols n0..n1-1 (the same as apply_equalizer on the whole stream, but only over this stretch)"""
    L = len(taps); m = L
    a, b = first + 2*n0 - m, first + 2*n1 + m
    seg = np.concatenate([np.zeros(max(0, -a), y2.dtype), y2[max(a, 0):min(b, len(y2))]])
    return apply_equalizer(seg, taps, bias, m, n1 - n0)

def calibrate(eq, start, nseg):
    """Scale and offset an equalized stream so the segment syncs are +5 -5 -5 +5. Returns (segments x 832, gain, offset)"""
    e = eq[start:start + nseg*SEG].reshape(nseg, SEG)
    hi, lo = e[:, 0].mean(), e[:, 1].mean()
    off, sc = (hi + lo)/2, (hi - lo)/10
    return (e - off)/sc, sc, off

def fit_levels(soft, iters=8):
    """Calibrate gain and offset from the data itself: the data are eight equally likely, equally spaced levels, so fit
    soft = gain*level + offset to the slicer's decisions and undo it, until it settles. (The sync symbols are a poor guide:
    the ISI that is left after equalizing biases them.) Returns the corrected array."""
    d = soft.copy()
    for _ in range(iters):
        v = d[:, 4:].ravel()[::31]
        g, o = np.polyfit(slice_levels(v), v, 1)
        d = (d - o)/g
    return d

def mer_of(d):
    """Error vector magnitude of the data symbols against the nearest level [dB]"""
    return 10*np.log10(21/np.mean((d - slice_levels(d))**2))

BLOCK_FIELDS = 10 # fields per equalizer block (~0.24 s; 6 s of channel 35: 76% correctable Reed-Solomon blocks with one equalizer, 89% with 40-field blocks, 97.6% with 10); 0 trains one for the whole capture

def recover(x, Fs, cf, lower, verbose=True, refine=2, block=BLOCK_FIELDS):
    """Capture -> equalized soft symbols. Returns a namespace:
       soft (segments x 832, levels: sync +-5, data -7..7), fs_seg (field sync segment's place in the 313), nseg,
       sym/y2/first/start/taps/bias/L (for retraining), pilot, ppm, mer_raw, mer_eq, ser, field_segments"""
    say = print if verbose else (lambda *a, **k: None)
    f_pilot, over = find_pilot(x, Fs, cf, lower)
    off = f_pilot - (lower + PILOT_OFFSET)
    say(f"Pilot at {f_pilot/1e6:.6f} MHz: {over:.1f} dB over the data, {off:+.0f} Hz from where it should be "
        f"({off/(lower + PILOT_OFFSET)*1e6:+.1f} ppm; the HackRF's clock and the broadcaster's together)")
    y, lock = to_symbol_rate_signal(x, Fs, cf, f_pilot)
    err = np.array(lock.err[20:])
    say(f"Pilot lock: phase error {np.std(err):.2f} rad rms over {len(err)} windows")
    step = Fs/(2*SYMBOL_RATE)
    pos0 = 1.0
    for it in range(8): # Clock error from the segment syncs, then again with it removed. A drift of 0.05 ppm still moves the
        # symbols by 0.3 of a symbol over a 0.6 s capture, so iterate until it is well under that
        y2 = matched(resample(y, step, pos0))
        prof, pos, seg = sync_fold(y2)
        slope = np.polyfit(np.arange(len(pos)), pos, 1)[0]/64 # Samples (of 2 per symbol) the sync drifts per segment, by regression
        ppm = slope/seg*1e6
        say(f"  pass {it}: sync drifts {slope:+.5f} samples/segment = clock {ppm:+.3f} ppm")
        if abs(ppm) < 0.003:
            break
        step *= 1 + slope/seg
    q = float(np.argmax(prof))
    shift = (q - 3) % 2 # Put the symbol instants (p-3, p-1, p+1, p+3 around the sync peak) on even samples
    if shift > 1:
        shift -= 2
    pos0 += shift*step
    y2 = matched(resample(y, step, pos0))
    prof, pos, seg = sync_fold(y2)
    q = float(np.argmax(prof))
    peak_sign = 1 if prof[int(q)] > 0 else -1
    first = int(round(q - 3)) % 2
    sym = y2[first::2]*peak_sign
    y2 = y2*peak_sign
    nseg = len(sym)//SEG
    fold = sym[:nseg*SEG].reshape(nseg, SEG).mean(0)
    pat = np.array([5, -5, -5, 5.0])
    match = np.array([np.dot(np.roll(fold, -s0)[:4], pat) for s0 in range(SEG)])
    start = int(np.argmax(match)) # The segment start, in symbols
    nseg = (len(sym) - start)//SEG
    segs, _, _ = calibrate(sym, start, nseg)
    data = segs[:, 4:]
    mer = 10*np.log10(21/np.mean((data - slice_levels(data))**2))
    say(f"{nseg} segments ({nseg*SEG/SYMBOL_RATE*1e3:.0f} ms); data MER {mer:.1f} dB unequalized "
        f"(8VSB needs about 15 dB; ISI from the channel is what an equalizer removes)")
    # Field sync: the field-sync segment repeats exactly 313 segments later, no other segment does
    A = data[:-FIELD]; B = data[FIELD:]
    rho = np.sum(A*B, 1)/np.sqrt(np.sum(A*A, 1)*np.sum(B*B, 1))
    hits = np.where(rho > 0.5)[0]
    res = SimpleNamespace(nseg=nseg, start=start, first=first, sym=sym, y2=y2, ppm=ppm, pilot=f_pilot, mer_raw=mer, rho=rho,
                          prof=prof, pos=pos, q=q, fs_seg=None, soft=segs, mer_eq=None, ser=None, L=96, taps=None, bias=0.0)
    if not len(hits):
        say("Field sync: none found")
        return res
    res.fs_seg = int(np.bincount(hits % FIELD).argmax())
    say(f"Field sync: {len(hits)} segments repeat 313 segments later (ordinary data gives ~{1/np.sqrt(828):.2f}); at segment "
        f"{res.fs_seg} of {FIELD}; {nseg/FIELD:.1f} fields in the capture, field time {FIELD*SEG/SYMBOL_RATE*1e3:.2f} ms (ATSC: 24.2 ms)")
    # Train an equalizer on the field syncs: their PN511 is the same in every field and is +-5, so the average over
    # fields, sliced to its sign, is the training sequence (no table of the sequence is needed)
    raw = sym[start:start + nseg*SEG].reshape(-1, SEG)
    fsegs = [k for k in range(len(raw) - 1) if k % FIELD == res.fs_seg]
    ntrain = 4 + 511
    avg = np.mean([raw[k][:ntrain] for k in fsegs], axis=0)
    known = np.sign(avg - np.mean(avg))*5.0
    # The channel moves (a long capture's taps differ ~30% from its first second's, and the Reed-Solomon failures sit in the stretches the static taps
    # fit worst), so the equalizer is trained per block of `block` fields, on that block's field syncs and then on its decided data symbols
    nb = max(1, int(round(nseg/FIELD/block))) if block and nseg/FIELD >= 1.5*block else 1
    edges = np.linspace(0, nseg, nb + 1).astype(int)
    parts, taps_all = [], []
    for bi in range(nb):
        s0, s1 = int(edges[bi]), int(edges[bi + 1])
        fb = [k for k in fsegs if s0 <= k < s1]
        idx = np.concatenate([start + k*SEG + np.arange(ntrain) for k in fb]); tgt = np.concatenate([known for _ in fb])
        taps, bias = train_equalizer(y2, first, idx, tgt, res.L)
        n0, n1 = start + s0*SEG, start + s1*SEG
        soft_b = fit_levels(calibrate(equalize_range(y2, taps, bias, first, n0, n1), 0, s1 - s0)[0])
        for it in range(refine): # decision-directed: the slicer's decisions on the data symbols are the target (its errors are noise to the fit, not bias)
            keep = np.array([k - s0 for k in range(s0, s1) if k % FIELD != res.fs_seg])
            flat = soft_b[keep, 4:].reshape(-1); pick = np.arange(0, len(flat), 3)
            datasyms = np.array([start + (s0 + k)*SEG + 4 + np.arange(828) for k in keep]).ravel()
            taps, bias = train_equalizer(y2, first, datasyms[pick], slice_levels(flat[pick]), res.L)
            soft_b = fit_levels(calibrate(equalize_range(y2, taps, bias, first, n0, n1), 0, s1 - s0)[0])
        parts.append(soft_b); taps_all.append((taps, bias))
    res.soft = np.concatenate(parts); res.taps, res.bias = taps_all[0]; res.blocks = nb
    res.mer_eq = mer_of(res.soft[:, 4:])
    say(f"Equalized ({res.L} taps, {nb} block(s) of ~{nseg/FIELD/nb:.0f} fields, trained on the field syncs then the data): data MER {res.mer_eq:.1f} dB")
    return res

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path")
    ap.add_argument("--fs", type=float, required=True, help="Sample rate [MS/s]")
    ap.add_argument("--cf", type=float, required=True, help="Capture centre [MHz]")
    ap.add_argument("--lower", type=float, required=True, help="Channel's lower edge [MHz]")
    ap.add_argument("--plot", metavar="PNG", help="Save a figure")
    a = ap.parse_args()
    Fs, cf, lower = a.fs*1e6, a.cf*1e6, a.lower*1e6
    x = np.fromfile(a.path, np.complex64)
    print(f"{a.path}: {len(x)/Fs:.2f} s at {Fs/1e6:g} MS/s, centre {cf/1e6:g} MHz, channel {a.lower:g}-{a.lower + 6:g} MHz")
    r = recover(x, Fs, cf, lower)
    if a.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2, 3, figsize=(16, 8), layout="constrained")
        n = 2**14
        sg = x[:len(x)//n*n].reshape(-1, n)*np.hanning(n)
        P = 10*np.log10(np.mean(np.abs(np.fft.fftshift(np.fft.fft(sg, axis=1), axes=1))**2, 0))
        fr = (np.fft.fftshift(np.fft.fftfreq(n, 1/Fs)) + cf)/1e6
        ax[0, 0].plot(fr, P, lw=.6); ax[0, 0].axvline(r.pilot/1e6, c="r", ls=":"); ax[0, 0].set_title("Capture: pilot (red)"); ax[0, 0].set_xlabel("MHz")
        ax[0, 1].plot(r.prof); ax[0, 1].set_title(f"Segment-sync correlation, folded on 832 symbols (peak {r.q:.0f})")
        ax[0, 2].plot(r.pos - r.pos[0]); ax[0, 2].set_title("Sync position drift after clock correction [samples]")
        raw = r.sym[r.start:r.start + r.nseg*SEG].reshape(-1, SEG)
        ax[1, 0].hist(calibrate(r.sym, r.start, r.nseg)[0][:, 4:].ravel(), bins=300, range=(-10, 10)); ax[1, 0].set_title(f"Data levels (MER {r.mer_raw:.1f} dB)")
        ax[1, 1].plot(r.rho); ax[1, 1].axhline(0.5, c="r", ls=":"); ax[1, 1].set_title("Segment s vs s+313: field syncs"); ax[1, 1].set_xlabel("segment")
        if r.mer_eq is not None:
            ax[1, 2].hist(r.soft[:, 4:].ravel(), bins=300, range=(-10, 10), color="C2"); ax[1, 2].set_title(f"After equalizing (MER {r.mer_eq:.1f} dB)")
            ax[0, 2].cla(); ax[0, 2].stem(np.arange(r.L)/2 - r.L/4, r.taps/np.max(np.abs(r.taps)), markerfmt="."); ax[0, 2].set_title("Equalizer taps (symbols)")
        fig.savefig(a.plot, dpi=75)
        print(f"Figure: {a.plot}")

if __name__ == "__main__":
    sys.exit(main())
