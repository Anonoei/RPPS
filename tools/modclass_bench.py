"""Modulation detection measured: the lowest Es/N0 each detector names each constellation at

    python tools/modclass_bench.py                                 # every modulation and detector, 100 trials a point
    python tools/modclass_bench.py --mods QPSK 8PSK --detectors cumulants-blind --trials 30
    python tools/modclass_bench.py --sweep -4:20:2 --mods 16QAM    # Pc at each Es/N0 instead of the threshold
    python tools/modclass_bench.py --jobs 8 --markdown --out thresholds.json
    python tools/modclass_bench.py --jobs 8 --update-baseline       # keep them in .benchmarks/ (--check compares)

Detectors:
    cumulants-blind   rpps.analysis.modulation.cumulants.classify(symbols): timed symbols, any carrier phase, the
                      noise share fitted. QAM and ASK orders do not separate blind (see classify), so for those a
                      family match counts and the threshold is marked *
    cumulants-aided   the same with esn0_db given (the true Es/N0): QAM and ASK orders separate
    cumulants-est     the same with esn0_db from the samples (rp.sync.quality.cyclic on RRC samples at 4 sps; symbols from
                      the matched filter at the known timing; no estimate, or a high one: blind)
    detect            rpps.sync.detect(samples, Fs, names): RRC (beta 0.35) at 4 samples a symbol, random timing, a
                      carrier offset of up to 1e-3 cycles a sample, the symbol rate searched blind
    detect-baud       the same with the symbol rate given (baud=): timing, carrier and modulation only

Trial i draws its symbols, carrier phase, timing, offset and a unit noise vector once from seed + i; every Es/N0 point
scales that same noise, so the shape of Pc against Es/N0 is the detector's, not the draw's. Pc is the share of --trials trials named right (a
point stops as soon as the target is met or missed). The threshold is the lowest Es/N0 (to --step dB) with Pc at least
--target at every point up to 20 dB: scanned down from 20 dB in 2 dB steps to the first that fails, then bisected
(the curves are not all monotone: a detector names its fallback right at low SNR); `>40` never reached it, -10 is the
floor. Es/N0 is per symbol at unit symbol energy.
detect scores against --names (the generated constellations, unit rms); a modulation outside them is n/a there.

Findings (2026-11-23, native kernels, 100 trials a point, Pc >= 0.9, 0.25 dB; cumulants 4096 symbols, detect 2048
at 4 sps; detect scoring each candidate at its own M2M4 Es/N0, the fit above 12.5 dB; the confusion is what the misses
were named just below the threshold):

    Es/N0 dB   blind    aided    est      detect   detect-baud   (detect-baud fitting each its own noise)   confused
    BPSK        6.50    -2.50    -6.25    -6.75    -7.50         (-0.50)                                    4ASK / 64QAM
    QPSK        6.75    -0.25     0.75     4.50     1.75         ( 4.75: 64QAM)                             16QAM
    8PSK        7.00     1.25     1.25     9.00     9.00         ( 9.25)                                    16PSK
    16PSK       7.50     7.50     7.75     8.25     8.25         ( 8.00)                                    8PSK
    16QAM       -0.50*   6.75     5.75    10.00    10.00         ( 9.75)                                    64QAM
    32QAM       6.50*    3.50     3.75     5.50     3.50         ( 5.25)                                    64QAM / 256QAM
    64QAM       2.25*    >40      >40     10.00    10.00         ( 4.50: the lean, not detection)           16QAM
    256QAM      0.00*    >40      >40      n/a      n/a                                                     32APSK
    16APSK      9.25     6.25     5.75     7.25     7.25         ( 9.00)                                    32QAM
    32APSK      8.75     4.50     5.00     8.00     8.00         ( 9.75, named right from -4 dB: the lean)  32QAM
    4ASK       22.00*    3.75     3.50     n/a      n/a                                                     BPSK / 8ASK
    8ASK        6.25*    4.50     8.00     n/a      n/a                                                     4ASK

- Blind cumulants need about 6.5-9 dB for any family; given the Es/N0, PSK and ASK go well below, and QAM orders
  separate, but 64QAM and 256QAM never reach 0.9 at 4096 symbols (64QAM levels at 0.64, read as 256QAM: their C42
  differ by 0.014, the scatter of a few thousand symbols' estimate). With the Es/N0 estimated from the samples (est,
  rp.sync.quality.cyclic), close to given.
- * the family only. Blind 4ASK hovers about 0.9 at every SNR above 8 dB (read as BPSK now and then): 22 is where
  this draw stopped failing, not a threshold.
- detect, each candidate fitting its own noise, leaned to the densest when the signal is weak (noise alone was
  32APSK, QPSK at 4 dB 64QAM): the generalised likelihood's bias. Each at its own moment Es/N0 it no longer does: BPSK
  7 dB lower, QPSK 3 with the rate given, 32QAM and the APSKs 1.75; 64QAM's honest threshold is 10, 16QAM level. In
  noise a front-end filter shaped (--frontend 0.8), Pc with the rate given 0.38 -> 0.55 at 4 dB, 0.75 -> 0.86 at 8
  dB. Tried and measured worse (scratch, 9 modulations x 20): one shared Es/N0 from rp.sync.quality.cyclic (as good
  on ideal RRC, but it reads a real transmitter's pulse low), the full GLRT (amplitude and noise free), a fit with
  signal + noise = power, a Kolmogorov-Smirnov distance on I and Q, an order penalty. Searching the rate costs QPSK
  2.75 dB: the rate search, not the modulation, limits it there.
- --confusion with --min-evidence 10: noise alone never named (it was named every time); wrong names at 8 dB: 8PSK as
  16PSK (7 in 50) and 16QAM as 64QAM (4 in 50), none at 12-16 dB; at 0-4 dB only 9-22% are named (the rest abstain).
- tests/unit/analysis/modulation/test_modclass_threshold.py keeps these (Pc >= 0.85 at 2 dB above, < 0.9 at 4 dB below).
"""
import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import rpps as rp  # noqa: E402
from rpps import symbol  # noqa: E402
from rpps.analysis.modulation import cumulants  # noqa: E402
from rpps.signal.filters.shaping import rrc_pulse  # noqa: E402

import _bench  # noqa: E402

DETECTORS = ("cumulants-blind", "cumulants-aided", "cumulants-est", "detect", "detect-baud")
MODS = ("BPSK", "QPSK", "8PSK", "16PSK", "16QAM", "32QAM", "64QAM", "256QAM", "16APSK", "32APSK", "4ASK", "8ASK")
NAMES = ("BPSK", "QPSK", "8PSK", "16PSK", "16QAM", "32QAM", "64QAM", "16APSK", "32APSK")
SHORT = {"cumulants-blind": "cb", "cumulants-aided": "ca", "cumulants-est": "ce", "detect": "d", "detect-baud": "db"}
NOISE = "noise" # a "modulation" of noise alone: every name a detector gives it is a false one
ABSTAIN = "-"   # what a detector that declines to name says
LO, HI, TOP = -10.0, 40.0, 20.0
FS, SPS, BETA, SPAN = 1e6, 4, 0.35, 12

def family(name):
    """QAM and ASK orders are one family to a blind cumulant match"""
    for f in ("QAM", "ASK"):
        if name.endswith(f):
            return f
    return name

def unit(points):
    p = np.asarray(points, dtype=np.complex128)
    return p/np.sqrt(np.mean(np.abs(p)**2))

def frontend(n, frac):
    """Noise through a front-end filter passing `frac` of Fs (26 dB down outside): the signal's band, and so what the
    matched filter sees, is unchanged; the band the signal leaves empty is not white any more"""
    if not frac or frac >= 1:
        return n
    N = np.fft.fft(n)
    N[np.abs(np.fft.fftfreq(len(n))) > frac/2] *= 0.05
    return np.fft.ifft(N)

def shape(s, sps, tau, beta=BETA, span=SPAN):
    """RRC samples of symbols s at sps (fractional timing tau in symbols), unit symbol energy"""
    n_sym = len(s)
    t = np.arange(int(n_sym*sps))/sps - tau
    k0 = np.floor(t).astype(int)
    x = np.zeros(len(t), complex)
    for d in range(-span, span + 1):
        k = k0 + d
        ok = (k >= 0) & (k < n_sym)
        x[ok] += s[k[ok]]*rrc_pulse(t[ok] - k[ok], beta)
    return x/np.sqrt(sps)

class Trials:
    """One modulation under one detector: the noiseless trials, drawn once, and Pc at any Es/N0"""
    def __init__(self, mod, detector, trials, symbols, seed, names=NAMES, front=None, min_evidence=None, min_gap=None):
        self.mod, self.detector, self.names = mod, detector, tuple(names)
        self.min_evidence, self.min_gap = min_evidence, min_gap
        self.blind = detector == "cumulants-blind"
        self.estimated = detector == "cumulants-est" # symbols at known timing, Es/N0 from the samples (rp.sync.quality.cyclic)
        self.sample = detector.startswith("detect") or self.estimated
        if mod == NOISE:
            pts = None
        else:
            pts = unit(symbol.load_or_generate(mod).points if detector.startswith("detect") else cumulants.candidates()[mod])
        self.sets = []
        for i in range(trials):
            rng = np.random.default_rng(seed + i)
            phase, tau, cfo = 2j*np.pi*rng.random(), rng.random(), rng.uniform(-1e-3, 1e-3)
            s = np.zeros(symbols, complex) if pts is None else pts[rng.integers(0, len(pts), symbols)]*np.exp(phase)
            if self.sample:
                x = shape(s, SPS, tau)
                if not self.estimated:
                    x = x*np.exp(2j*np.pi*cfo*np.arange(len(x)))
            else:
                x = s
            n = (rng.standard_normal(len(x)) + 1j*rng.standard_normal(len(x)))/np.sqrt(2)
            if self.sample:
                n = frontend(n, front)
            self.sets.append((x, n, tau))
        self.confused = {}

    @property
    def applies(self):
        return not self.detector.startswith("detect") or self.mod in self.names or self.mod == NOISE

    def symbols(self, y, tau):
        """The matched filter's symbols at the known timing"""
        from rpps.sync import acquire
        c = acquire.centres_in(len(y), tau*SPS + np.ceil((SPAN*SPS + 5 - tau*SPS)/SPS)*SPS, SPS, SPAN)
        return acquire.matrix_mf(y, c, SPS, BETA, SPAN)

    def named(self, i, esn0):
        """What the detector calls trial i at esn0"""
        x, n, tau = self.sets[i]
        y = x + np.sqrt(10**(-esn0/10))*n # per-sample noise variance 10^(-Es/N0/10): Es/N0 at unit symbol energy
        if self.estimated:
            est = rp.sync.quality.cyclic(y, SPS)
            usable = est is not None and np.isfinite(est.esn0_db)
            return self._classify(self.symbols(y, tau), est.esn0_db if usable else None)
        if self.sample:
            kw = {} if self.min_evidence is None else {"min_evidence": self.min_evidence}
            try:
                got = rp.sync.detect(y.astype(np.complex64), FS, names=self.names,
                                     baud=FS/SPS if self.detector == "detect-baud" else None, **kw).mod
            except rp.errors.NoLock:
                return ABSTAIN
            except Exception as e: # a miss: too few symbols, ...
                return type(e).__name__
            return ABSTAIN if got is None else got
        return self._classify(y, None if self.blind else esn0)

    def _classify(self, syms, esn0_db):
        kw = {} if self.min_gap is None else {"min_gap": self.min_gap}
        r = cumulants.classify(syms, esn0_db=esn0_db, **kw)
        return r[0].name if r else ABSTAIN

    def right(self, got):
        if self.mod == NOISE:
            return got == ABSTAIN
        if self.blind:
            return family(got) == family(self.mod)
        return got == self.mod

    def pc(self, esn0, target=None):
        """Share named right; with a target, stops once it is met or missed (the count then reaches past it)"""
        T = len(self.sets)
        need = None if target is None else int(np.ceil(target*T))
        ok = bad = 0
        wrong = {}
        for i in range(T):
            got = self.named(i, esn0)
            if self.right(got):
                ok += 1
            else:
                bad += 1
                wrong[got] = wrong.get(got, 0) + 1
            if need is not None and (ok >= need or bad > T - need):
                break
        self.confused[esn0] = wrong
        return ok/T if need is None else (1.0 if ok >= need else 0.0), ok, ok + bad

    def threshold(self, target, step, top=TOP, grid=2.0):
        """Lowest Es/N0 from which Pc stays at least `target` up to `top`: scanned down from `top` on `grid` until a
        point fails, then bisected between it and the last that passed. A detector that falls back on one candidate when
        the signal is weak (rp.sync.detect picks the densest) names that one right at low SNR and wrong above it: the
        scan does not count that. Not reached at `top`: up in grid steps to HI, else None"""
        passes = lambda e: self.pc(e, target)[0] >= 1.0 # noqa: E731
        hi = top
        while not passes(hi):
            if hi >= HI:
                return None
            hi = min(hi + grid, HI)
        lo = hi - grid
        while passes(lo):
            if lo <= LO:
                return LO
            lo, hi = max(lo - grid, LO), lo
        while hi - lo > step:
            mid = (lo + hi)/2
            if passes(mid):
                hi = mid
            else:
                lo = mid
        return hi

def job(args):
    mod, det, a = args
    t0 = time.perf_counter()
    syms = a["symbols"] or (2048 if det.startswith("detect") else 4096)
    tr = Trials(mod, det, a["trials"], syms, a["seed"], a["names"], a["frontend"], a["min_evidence"], a["min_gap"])
    if not tr.applies:
        return mod, det, "n/a", None, 0.0
    if a["confusion"]:
        out = {}
        for k, e in enumerate(a["confusion"]):
            if mod == NOISE and k: # noise alone is the same at every point (detectors normalize)
                out[e] = out[a["confusion"][0]]
                continue
            named = {}
            for i in range(len(tr.sets)):
                g = tr.named(i, e)
                named[g] = named.get(g, 0) + 1
            out[e] = named
        return mod, det, "confusion", out, time.perf_counter() - t0
    if a["sweep"]:
        curve = []
        for e in a["sweep"]:
            p, _, _ = tr.pc(e)
            top = max(tr.confused[e].items(), key=lambda kv: kv[1])[0] if tr.confused[e] else ""
            curve.append((e, p, top))
        return mod, det, "sweep", curve, time.perf_counter() - t0
    thr = tr.threshold(a["target"], a["step"])
    below = thr - a["step"] if thr is not None and thr > LO else None
    conf = {}
    if below is not None:
        tr.pc(below)
        conf = tr.confused[below]
    return mod, det, thr, conf, time.perf_counter() - t0

def confusion(res, a, rec):
    """Per detector and Es/N0: sent -> named counts, each name's false-discovery rate, what noise alone is called"""
    for d in a.detectors:
        rows = {m: extra for m, dd, thr, extra, _ in res if dd == d and thr == "confusion"}
        if not rows:
            continue
        for e in a.confusion:
            M = {m: rows[m][e] for m in rows}
            names = sorted({g for c in M.values() for g in c}, key=lambda g: (g == ABSTAIN, g))
            w = max(6, max(len(g) for g in names) + 1)
            print(f"\n{d} at {e:g} dB: sent (rows) named (columns), {a.trials} trials")
            print(" "*8 + "".join(g.rjust(w) for g in names))
            for m, c in M.items():
                print(m.ljust(8) + "".join((str(c.get(g, 0)) if c.get(g) else ".").rjust(w) for g in names))
            fdr = []
            for g in names:
                if g == ABSTAIN:
                    continue
                total = sum(c.get(g, 0) for c in M.values())
                wrong = sum(c.get(g, 0) for m, c in M.items() if not _same(d, m, g))
                if total:
                    fdr.append(f"{g} {wrong/total:.2f}")
                    rec.add(f"{d} FDR {g} @{e:g} dB", wrong/total, "", "lower", tol=0.0)
            named = sum(c.get(g, 0) for m, c in M.items() for g in c if g != ABSTAIN and m != NOISE)
            sent = sum(sum(c.values()) for m, c in M.items() if m != NOISE)
            print(f"  false discovery: {', '.join(fdr)}")
            print(f"  named {named}/{sent} sent", end="")
            if NOISE in M:
                n = M[NOISE]
                fa = sum(v for g, v in n.items() if g != ABSTAIN)/max(sum(n.values()), 1)
                print(f"; noise alone named {fa:.2f} of the time", end="")
                rec.add(f"{d} noise named", fa, "", "lower", tol=0.0)
            print()

def _same(det, sent, named):
    if sent == NOISE:
        return False
    if det == "cumulants-blind":
        return family(sent) == family(named)
    return sent == named

def cell(thr, mod, det):
    if thr == "n/a":
        return "n/a"
    if thr is None:
        return f">{HI:g}"
    star = "*" if det == "cumulants-blind" and family(mod) != mod else ""
    return f"{thr:.2f}{star}"

def frange(spec):
    a, b, s = (float(v) for v in spec.split(":"))
    return [round(v, 6) for v in np.arange(a, b + s/2, s)]

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--mods", nargs="+", default=list(MODS), help=f"of {list(MODS)}")
    ap.add_argument("--detectors", nargs="+", default=list(DETECTORS), choices=DETECTORS)
    ap.add_argument("--names", nargs="+", default=list(NAMES), help="detect's candidate constellations")
    ap.add_argument("--target", type=float, default=0.9, help="Pc the threshold must reach")
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--symbols", type=int, default=0, help="a trial's symbols (default 4096 cumulants, 2048 detect)")
    ap.add_argument("--step", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--sweep", type=frange, help="a:b:step Es/N0 dB: Pc at each point, no threshold search")
    ap.add_argument("--confusion", type=frange, help="a:b:step Es/N0 dB: what each modulation (and noise alone) is "
                    "named, the false-discovery rate of each name")
    ap.add_argument("--frontend", type=float, help="noise through a front-end filter passing this share of Fs")
    ap.add_argument("--min-evidence", type=float, help="detect's min_evidence (abstain below it)")
    ap.add_argument("--min-gap", type=float, help="cumulants.classify's min_gap (abstain below it)")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--markdown", action="store_true")
    ap.add_argument("--out", type=Path, help="JSON results")
    _bench.add_args(ap)
    a = ap.parse_args(argv)
    opts = {k: getattr(a, k) for k in ("trials", "symbols", "seed", "names", "target", "step", "sweep", "confusion",
                                         "frontend", "min_evidence", "min_gap")}
    mods = list(a.mods) + ([NOISE] if a.confusion and NOISE not in a.mods else [])
    work = [(m, d, opts) for m in mods for d in a.detectors]
    what = "confusion" if a.confusion else "Pc sweep" if a.sweep else f"Es/N0 dB for Pc >= {a.target:g}"
    print(f"{what}; {a.trials} trials a point, step {a.step} dB, seed {a.seed}", flush=True)
    if a.jobs > 1:
        with ProcessPoolExecutor(a.jobs) as ex:
            res = list(ex.map(job, work))
    else:
        res = []
        for w in work:
            res.append(job(w))
            m, d, thr, extra, took = res[-1]
            if not (a.sweep or a.confusion):
                print(f"  {m:7s} {d:16s} {cell(thr, m, d):>8s}   ({took:.0f} s)", flush=True)
    R = {(m, d): (thr, extra) for m, d, thr, extra, _ in res}
    rec = _bench.Record("modclass")
    if a.confusion:
        confusion(res, a, rec)
        return _bench.finish(a, rec)
    for m, d, thr, extra, _ in res:
        if thr == "n/a":
            continue
        if a.sweep: # fixed seeds: the same detector names the same trials, any change is the detector's
            for e, p, _top in extra:
                rec.add(f"{m} {d} Pc @{e:g} dB", p, "", "higher", tol=0.0)
        else:
            rec.add(f"{m} {d}", thr, "dB", "lower", tol=a.step)
    if a.sweep:
        for m, d, thr, curve, _ in res:
            if thr == "n/a":
                continue
            print(f"\n{m} {d}\n  Es/N0     Pc   most often instead")
            for e, p, top in curve:
                print(f"  {e:6.2f}  {p:5.2f}   {top}")
    else:
        w = max(len(d) for d in a.detectors)
        print("\n" + "mod".ljust(8) + "".join(d.rjust(w + 2) for d in a.detectors) + "   confused (just below)")
        for m in a.mods:
            conf = []
            for d in a.detectors:
                thr, c = R[(m, d)]
                if c:
                    conf.append(f"{SHORT[d]}:{max(c, key=c.get)}")
            print(m.ljust(8) + "".join(cell(R[(m, d)][0], m, d).rjust(w + 2) for d in a.detectors) + "   " + " ".join(conf))
        if a.markdown:
            print("\n| Modulation | " + " | ".join(a.detectors) + " |\n|---|" + "---|"*len(a.detectors))
            for m in a.mods:
                print(f"| {m} | " + " | ".join(cell(R[(m, d)][0], m, d) for d in a.detectors) + " |")
    if a.out:
        out = [{"mod": m, "detector": d, "threshold" if not a.sweep else "curve": thr if not a.sweep else extra,
                "confused": extra if not a.sweep else None} for m, d, thr, extra, _ in res]
        a.out.write_text(json.dumps({"args": {k: v for k, v in opts.items()}, "results": out}, indent=1, default=str))
    return _bench.finish(a, rec)

if __name__ == "__main__":
    sys.exit(main())
