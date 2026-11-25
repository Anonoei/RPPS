"""Signal finders scored against synthetic scenes with known signals: detection, false boxes, box placement, SNR, time

    python tools/finder_bench.py                       # every effort level, every scene kind, 8 seeds
    python tools/finder_bench.py --seeds 32 --kinds dense pedestal
    python tools/finder_bench.py --finders balanced    # one effort level of rp.analysis.find_signals
    python tools/finder_bench.py --show dense 3        # plot one scene with its truth and each level's boxes

A scene is complex noise whose spectrum is the sum of its signals' spectra over the noise floor (one inverse FFT: a
signal's samples are Gaussian with its pulse's power spectrum, which is all an energy detector sees of it), times the
receiver's passband (roll-off at the band edges, tilt, ripple), plus what is gated in time (bursts) and lines (tones, a
DC spike). The kinds follow what reviewers marked wrong on real satellite captures (rpps-06's 518 reviewed boxes):

    sparse     1-10 RRC carriers (beta 0.05-0.5), 30 kHz-8 MHz wide at 50 MS/s, apart, in-band SNR -6..25 dB
    dense      3-7 carriers 1.0-1.5 widths apart, levels 3-6 dB apart (the "two signals in one" boxes)
    weak       1-5 dB carriers between 20 dB neighbours (a neighbour's skirt is not the floor)
    pedestal   a 10-20 dB transponder hump tens of MHz wide with carriers on it (slices of it are false boxes)
    humps      two broad humps with a V between them, and a sloped passband (no carriers: every box is false)
    bursts     TDMA bursts at duty 0.05-0.5 and a 20 MHz TDD carrier (slices of it are false boxes)
    tones      CW lines and a DC spike (the spike is not a signal)
    edge       a carrier cut by the band edge (Signal.clipped)
    noise      noise with the passband's shape only (every box is false)

Scores (per finder, per kind):
    Pd by in-band SNR [dB, signal power over the noise in (1+beta) x baud]: a truth is found when a box overlaps it with
        frequency IoU >= 0.5 (a tone: the box holds it and is under 10 bins + 2% of the band)
    false: boxes matching no truth (boxes over an `ignore` truth, a pedestal or the DC spike, count neither way)
    merged: boxes holding the centres of two or more truths
    centre: |box centre - truth| / truth baud, width: box width / baud (reviewers' -10 dB width of an RRC carrier is
        1.15-1.19 baud), esn0: box snr + 10 log10(width / baud) - true Es/N0 [dB]
    time IoU: for bursts, when the finder gives t0/t1
    ms: the finder's time for the scene
"""
import argparse
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

FS = 50e6

@dataclass
class Truth:
    f: float          # centre [Hz from the capture's centre]
    baud: float       # symbol rate [Hz] (a tone: 0)
    beta: float = 0.35
    snr: float = 0.0  # in-band SNR [dB] over (1+beta) baud
    t0: float = 0.0   # on [s from the start]
    t1: float = None  # off (None: the whole capture)
    kind: str = "rrc" # rrc, flat (steep-edged block), tone
    ignore: bool = False # a pedestal or a spur: no box need be found, a box on it is not false

    @property
    def bw(self):
        return (1 + self.beta)*self.baud if self.kind != "tone" else 0.0

    @property
    def lo(self):
        return self.f - self.bw/2

    @property
    def hi(self):
        return self.f + self.bw/2

    @property
    def esn0(self):
        return self.snr + 10*np.log10(1 + self.beta)

@dataclass
class Scene:
    kind: str
    x: np.ndarray
    Fs: float
    truth: list = field(default_factory=list)

def rc_psd(f, baud, beta):
    """A raised-cosine power spectrum (an RRC carrier's), unit area"""
    a = np.abs(f)
    lo, hi = (1 - beta)*baud/2, (1 + beta)*baud/2
    p = np.where(a <= lo, 1.0, 0.0)
    if beta > 0:
        t = (a > lo) & (a <= hi)
        p = np.where(t, 0.5*(1 + np.cos(np.pi/(beta*baud)*(a - lo))), p)
    return p/baud

def flat_psd(f, baud, beta):
    """A steep-edged block (flat top), unit area over (1+beta) baud, edges a few % of its width"""
    w = (1 + beta)*baud
    e = 0.02*w
    return np.clip((w/2 - np.abs(f))/e + 0.5, 0, 1)/w

def passband(f, Fs, rng, tilt_db=None, ripple_db=0.5):
    """The receiver's gain [linear power]: roll-off in the outer 4% each side, a tilt, slow ripple"""
    a = np.abs(f)/(Fs/2)
    roll = np.where(a > 0.92, 10**(-40*((a - 0.92)/0.08)**2/10), 1.0)
    tilt = rng.uniform(-2, 2) if tilt_db is None else tilt_db
    rip = ripple_db*np.sin(2*np.pi*f/Fs*rng.uniform(1.5, 4) + rng.uniform(0, 6.3))
    return roll*10**((tilt*f/(Fs/2) + rip)/10)

def synth(rng, N, Fs, truth, extra_psd=None, tilt_db=None):
    """Samples of a scene: noise (unit power before the passband) + continuous carriers in one inverse FFT, bursts
    gated in time, tones added as lines. extra_psd(f) [power/Hz, over the noise floor] is anything without a truth box
    (a pedestal, humps)"""
    f = np.fft.fftfreq(N, 1/Fs)
    n0 = 1/Fs # noise PSD: unit power over Fs
    S = np.full(N, n0)
    if extra_psd is not None:
        S = S + n0*extra_psd(f)
    gated = []
    for t in truth:
        if t.kind == "tone" or t.ignore: # (an ignored band is drawn by extra_psd, an ignored tone below)
            continue
        P = 10**(t.snr/10)*n0*t.bw # signal power for its in-band SNR
        shape = rc_psd if t.kind == "rrc" else flat_psd
        psd = P*shape(f - t.f, t.baud, t.beta)
        if t.t1 is None and t.t0 == 0:
            S = S + psd
        else:
            gated.append((t, psd))
    H = passband(f, Fs, rng, tilt_db)
    g = (rng.standard_normal(N) + 1j*rng.standard_normal(N))/np.sqrt(2)
    x = np.fft.ifft(g*np.sqrt(S*H*Fs))*np.sqrt(N)
    tt = np.arange(N)/Fs
    for t, psd in gated:
        g = (rng.standard_normal(N) + 1j*rng.standard_normal(N))/np.sqrt(2)
        y = np.fft.ifft(g*np.sqrt(psd*H*Fs))*np.sqrt(N)
        ramp = max(1, int(4*Fs/t.baud)) # a few symbols of ramp, as a burst's power amplifier does
        on = ((tt >= t.t0) & (tt < (t.t1 if t.t1 is not None else np.inf))).astype(float)
        on = np.convolve(on, np.hanning(2*ramp)/np.hanning(2*ramp).sum(), mode="same")
        x = x + y*on
    for t in truth:
        if t.kind == "tone":
            # A tone's in-band SNR: its power over the noise in one Hann bin at nfft 4096 (what a survey sees)
            P = 10**(t.snr/10)*n0*Fs/4096
            x = x + np.sqrt(P)*np.exp(2j*np.pi*(t.f*tt + rng.uniform()))
    return x.astype(np.complex64)

def _place(rng, Fs, widths, gap=(0.2, 3.0), span=0.85):
    """Centres for carriers of these widths, left to right with gaps of `gap` x the larger width, inside `span` of the
    band; fewer if they don't fit"""
    out, lo = [], -span*Fs/2 + rng.uniform(0, 0.1)*Fs
    for w in widths:
        c = lo + w/2
        if c + w/2 > span*Fs/2:
            break
        out.append(c)
        lo = c + w/2 + rng.uniform(*gap)*w
    return out

def _baud(rng, lo=30e3, hi=8e6):
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))

def scene(kind, rng, N=1 << 20, Fs=FS):
    T = N/Fs
    truth, extra, tilt = [], None, None
    if kind == "sparse":
        bauds = [_baud(rng) for _ in range(rng.integers(1, 11))]
        widths = [(1 + 0.35)*b for b in bauds]
        for b, c in zip(bauds, _place(rng, Fs, widths)):
            truth.append(Truth(c, b, rng.uniform(0.05, 0.5), rng.uniform(-6, 25)))
    elif kind == "dense":
        for _ in range(rng.integers(1, 3)):
            b = _baud(rng, 100e3, 3e6)
            beta = rng.uniform(0.2, 0.35)
            n = int(rng.integers(3, 8))
            start = rng.uniform(-0.4, 0.4 - n*1.5*(1 + beta)*b/Fs)*Fs
            if start + n*1.5*(1 + beta)*b > 0.42*Fs:
                continue
            c, lvl = start, rng.uniform(8, 20)
            for k in range(n):
                if any(abs(c - t.f) < (t.bw + (1 + beta)*b)/2 for t in truth):
                    break
                truth.append(Truth(c, b, beta, lvl))
                c += rng.uniform(1.0, 1.5)*(1 + beta)*b
                lvl = float(np.clip(lvl + rng.choice([-1, 1])*rng.uniform(3, 6), 4, 25))
    elif kind == "weak":
        b = _baud(rng, 200e3, 2e6)
        w = 1.35*b
        c = rng.uniform(-0.3, 0.2)*Fs
        for k, snr in enumerate([20, rng.uniform(1, 5), 20]):
            truth.append(Truth(c + k*rng.uniform(1.1, 1.4)*w, b, 0.35, snr if k != 1 else float(snr)))
    elif kind == "pedestal":
        width, h = rng.uniform(20e6, 34e6), rng.uniform(10, 20)
        c0 = rng.uniform(-0.1, 0.1)*Fs
        extra = lambda f: (10**(h/10) - 1)*flat_psd(f - c0, width/1.35, 0.35)*width
        truth.append(Truth(c0, width/1.35, 0.35, h, kind="flat", ignore=True))
        bauds = [_baud(rng, 100e3, 3e6) for _ in range(rng.integers(2, 7))]
        cs = _place(rng, width, [1.35*b for b in bauds], gap=(0.3, 2.0), span=0.8)
        for b, c in zip(bauds, cs):
            # Its SNR over the pedestal (the local floor): the pedestal raises the noise under it by h
            truth.append(Truth(c0 + c, b, 0.35, rng.uniform(4, 20) + h))
        for t in truth[1:]:
            t.snr_local = t.snr - h
    elif kind == "humps":
        c1, c2 = rng.uniform(-0.3, -0.1)*Fs, rng.uniform(0.1, 0.3)*Fs
        s1, s2 = rng.uniform(3e6, 8e6), rng.uniform(3e6, 8e6)
        h1, h2 = rng.uniform(3, 8), rng.uniform(3, 8)
        extra = lambda f: (10**(h1/10) - 1)*np.exp(-((f - c1)/s1)**2) + (10**(h2/10) - 1)*np.exp(-((f - c2)/s2)**2)
        tilt = rng.choice([-1, 1])*rng.uniform(3, 6)
    elif kind == "bursts":
        b = _baud(rng, 50e3, 1e6)
        c = rng.uniform(-0.35, -0.05)*Fs
        slot = T/8
        for k in range(8):
            if rng.uniform() < 0.5:
                d = rng.uniform(0.3, 0.9)*slot
                truth.append(Truth(c, b, 0.35, rng.uniform(6, 20), t0=k*slot, t1=k*slot + d))
        if rng.uniform() < 0.7: # a TDD carrier: 20 MHz, on for 10-50% of each frame
            bw, duty = 20e6/1.1, rng.uniform(0.1, 0.5)
            ct = rng.uniform(0.12, 0.2)*Fs
            for k in range(4):
                truth.append(Truth(ct, bw, 0.1, rng.uniform(8, 15), kind="flat", t0=k*T/4, t1=k*T/4 + duty*T/4))
    elif kind == "tones":
        for c in _place(rng, Fs, [Fs/50]*int(rng.integers(1, 6)), gap=(1, 4)):
            truth.append(Truth(c, 0.0, 0.0, rng.uniform(6, 30), kind="tone"))
        truth.append(Truth(0.0, 0.0, 0.0, 25.0, kind="tone", ignore=True)) # the DC spike
    elif kind == "edge":
        b = _baud(rng, 1e6, 6e6)
        side = rng.choice([-1, 1])
        truth.append(Truth(side*(0.46*Fs), b, 0.25, rng.uniform(10, 20)))
        b2 = _baud(rng, 200e3, 2e6)
        truth.append(Truth(rng.uniform(-0.2, 0.2)*Fs, b2, 0.35, rng.uniform(6, 20)))
    elif kind == "noise":
        tilt = rng.uniform(-3, 3)
    else:
        raise ValueError(kind)
    truth = [t for t in truth if t.kind == "tone" or (abs(t.f) - t.bw/2 < 0.5*Fs)]
    return Scene(kind, synth(rng, N, Fs, truth, extra, tilt), Fs, truth)

KINDS = ("sparse", "dense", "weak", "pedestal", "humps", "bursts", "tones", "edge", "noise")

# --- finders: callable(x, Fs) -> list of boxes with lo, hi, mid, width, snr (and t0, t1 when they give time)

def _finder(level, segment="stair"):
    def run(x, Fs):
        import rpps as rp
        from rpps.analysis.spectrum.find import find
        with rp.effort(level):
            return list(find(x, Fs, segment=segment))
    run.__name__ = level if segment == "stair" else f"{segment}-{level}"
    return run

FINDERS = {lv: _finder(lv) for lv in ("fast", "balanced", "max")}
FINDERS.update({f"net-{lv}": _finder(lv, "net") for lv in ("fast", "balanced")}) # (the learned segmenter)

# --- scoring

def _iou(a0, a1, b0, b1):
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    union = max(a1, b1) - min(a0, b0)
    return inter/union if union > 0 else 0.0

def _matches(box, t, Fs):
    if t.kind == "tone":
        return box.lo - Fs/4096 <= t.f <= box.hi + Fs/4096 and box.width <= 10*Fs/4096 + 0.02*Fs
    lo, hi = max(t.lo, -Fs/2), min(t.hi, Fs/2) # what of it is in the band
    return _iou(box.lo, box.hi, lo, hi) >= 0.5

def score(sc, boxes, T):
    """Per-truth and per-box outcomes of one scene"""
    rows, used = [], set()
    # Truths on at the same frequency (bursts of one carrier) are one frequency truth; time is scored per burst
    for t in sc.truth:
        cand = [i for i, b in enumerate(boxes) if _matches(b, t, sc.Fs)]
        hit = None
        if cand:
            # the box whose span is closest to it (bursts: also overlapping in time when boxes have time)
            def key(i):
                b = boxes[i]
                tiou = _tiou(b, t, T)
                return (-tiou, abs(b.mid - t.f))
            hit = min(cand, key=key)
            used.update(cand)
        rows.append((t, boxes[hit] if hit is not None else None))
    false = [b for i, b in enumerate(boxes) if i not in used and not any(
        t.ignore and (_iou(b.lo, b.hi, t.lo, t.hi) > 0.3 or (t.kind == "tone" and b.lo <= t.f <= b.hi)) for t in sc.truth)]
    merged = [b for b in boxes if not getattr(b, "holds", 0) and
              len({round(t.f) for t in sc.truth if not t.ignore and b.lo < t.f < b.hi}) >= 2]
    return rows, false, merged

def _tiou(b, t, T):
    """A box's time overlap with a truth: the best of its bursts (or its t0..t1; 1 when it gives no time)"""
    spans = getattr(b, "bursts", None) or ([(b.t0, b.t1)] if getattr(b, "t1", None) is not None else None)
    if spans is None:
        return 1.0
    return max(_iou(a, z, t.t0, t.t1 if t.t1 is not None else T) for a, z in spans)

BINS = [(-99, 0), (0, 3), (3, 6), (6, 10), (10, 20), (20, 99)]

def bench(finders, kinds, seeds, N):
    res = {}
    for kind in kinds:
        for seed in range(seeds):
            sc = scene(kind, np.random.default_rng(1000*seed + KINDS.index(kind)), N)
            T = len(sc.x)/sc.Fs
            for name, fn in finders.items():
                t0 = time.perf_counter()
                boxes = fn(sc.x, sc.Fs)
                ms = 1e3*(time.perf_counter() - t0)
                rows, false, merged = score(sc, boxes, T)
                r = res.setdefault((name, kind), dict(found={}, n={}, false=0, merged=0, centre=[], width=[], esn0=[],
                                                      tiou=[], ms=[], scenes=0))
                r["scenes"] += 1
                r["false"] += len(false)
                r["merged"] += len(merged)
                r["ms"].append(ms)
                for t, b in rows:
                    if t.ignore:
                        continue
                    snr = getattr(t, "snr_local", t.snr)
                    k = next(i for i, (a, z) in enumerate(BINS) if a <= snr < z)
                    r["n"][k] = r["n"].get(k, 0) + 1
                    if b is None:
                        continue
                    r["found"][k] = r["found"].get(k, 0) + 1
                    if t.kind == "rrc" and abs(t.f) + t.bw/2 < 0.45*sc.Fs:
                        r["centre"].append(abs(b.mid - t.f)/t.baud)
                        r["width"].append(b.width/t.baud)
                        if b.snr is not None:
                            r["esn0"].append(b.snr + 10*np.log10(b.width/t.baud) - (t.esn0 - (t.snr - snr)))
                    if t.t1 is not None and getattr(b, "t1", None) is not None:
                        r["tiou"].append(_tiou(b, t, T))
    return res

def report(res, finders, kinds):
    head = "Pd " + " ".join(f"{a if a > -99 else '':>3}..{z if z < 99 else '':<3}" for a, z in BINS)
    print(f"{'finder':<14}{'kind':<10}{head}  false merged  centre/baud  width/baud(10-90%)  esn0 err(med,90%)  tIoU   ms")
    for kind in kinds:
        for name in finders:
            r = res.get((name, kind))
            if r is None:
                continue
            pd = " ".join(f"{r['found'].get(k, 0)/r['n'][k]:>8.2f}" if r["n"].get(k) else f"{'-':>8}" for k in range(len(BINS)))
            c = f"{np.median(r['centre']):.3f}" if r["centre"] else "-"
            w = (f"{np.median(r['width']):.2f} ({np.percentile(r['width'], 10):.2f}-{np.percentile(r['width'], 90):.2f})"
                 if r["width"] else "-")
            e = (f"{np.median(r['esn0']):+.2f}, {np.percentile(np.abs(r['esn0']), 90):.2f}" if r["esn0"] else "-")
            ti = f"{np.median(r['tiou']):.2f}" if r["tiou"] else "-"
            print(f"{name:<14}{kind:<10}   {pd}  {r['false']:>5} {r['merged']:>6}  {c:>11}  {w:>18}  {e:>17}  {ti:>4}  "
                  f"{np.median(r['ms']):6.1f}")
        print()

def show(kind, seed, finders, N):
    import matplotlib.pyplot as plt
    sc = scene(kind, np.random.default_rng(1000*seed + KINDS.index(kind)), N)
    from rpps.analysis.spectrum.signal import welch
    p, _ = welch(sc.x, sc.Fs, 4096)
    f = (np.arange(4096) - 2048)*sc.Fs/4096
    fig, axes = plt.subplots(len(finders), 1, sharex=True, figsize=(14, 3*len(finders)), squeeze=False)
    for ax, (name, fn) in zip(axes[:, 0], finders.items()):
        ax.plot(f/1e6, 10*np.log10(p), lw=0.6, color="k")
        for t in sc.truth:
            ax.axvspan(t.lo/1e6, max(t.hi, t.lo + sc.Fs/4096)/1e6, color="grey" if t.ignore else "g", alpha=0.2)
        for b in fn(sc.x, sc.Fs):
            ax.axvspan(b.lo/1e6, b.hi/1e6, ymin=0.9, ymax=1.0, color="r", alpha=0.6)
        ax.set_title(name)
    axes[-1, 0].set_xlabel("MHz")
    plt.show()

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--kinds", nargs="*", default=list(KINDS))
    ap.add_argument("--finders", nargs="*", default=None)
    ap.add_argument("-N", type=int, default=1 << 20, help="samples a scene")
    ap.add_argument("--show", nargs=2, metavar=("KIND", "SEED"))
    a = ap.parse_args()
    finders = {k: v for k, v in FINDERS.items() if a.finders is None or k in a.finders}
    if a.show:
        show(a.show[0], int(a.show[1]), finders, a.N)
        return
    report(bench(finders, a.kinds, a.seeds, a.N), finders, a.kinds)

if __name__ == "__main__":
    main()
