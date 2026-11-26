"""Measure a receiver's own errors from signals of known frequency or rate: its reference (ppm) and an LNB's LO

    python projects/reference/frequency.py --freq 751 --fs 15.36                 # the radio (RPPS_RADIO, PYSDRLIB_REMOTE)
    python projects/reference/frequency.py --file cap.cf32 --fmt cf32 --fs 15.36 --freq 751

Prints what it measured and the pysdrlib calls that keep it: `set_freq_correction(ppm, accuracy=)` for the radio's
reference, and a chain's `lo` and `lo_accuracy` (a frontends file entry) for an LNB. RPPS reads them back from the
receive context (`RxContext`: `ppm_accuracy`, `chain.lo_accuracy`); measuring them is this project's, not the library's.

A capture's frequencies are off by the receive chain's errors, which the samples alone can't tell from the signal's:

    the radio's reference, `ppm` (positive: it runs fast): every frequency the radio tunes and its sample clock are off by
        it. A carrier at IF f reads -f ppm 1e-6 off (through an LO below the signal; above it, the sign turns), and
        every measured rate reads 1/(1 + ppm 1e-6) of the true one
    an LNB's LO, `lo_error` [Hz] (positive: high): every carrier through it reads that much low, either side of it

So a carrier at a known frequency gives lo_error + its IF times ppm, and a known rate gives ppm alone; together, or
carriers at different IFs, both. `fit` solves for them by least squares with priors (a radio's reference within
`RADIO_PPM`, a disciplined one within `DISCIPLINED_PPM`, an LNB's LO within `LNB_HZ`: what they are when nothing
measures them), and `Calibration` corrects any frequency or rate of the capture, with its standard error.

References:
    FrequencyReference   a carrier over the air at a known frequency (`true`), or on a raster (`raster`: LTE's 100 kHz):
                         a raster is resolved only once the rest pins the error under a quarter of its step
    RateReference        a clock error measured against a known rate (`rate(measured, nominal)`: ATSC's 10.762238 MBd,
                         an LTE half-frame's 5 ms)
    lte(x, Fs, offset)   both, from an LTE cell's PSS and SSS (rpps.grid.lte.LteCell): the carrier on its 100 kHz raster,
                         the sample clock against its 5 ms half-frames. The base stations are GPS-locked (0.05 ppm)
    find, calibrate      the references every LTE-wide signal in a capture gives; a source's Calibration from them

On air (2026-11-26): a HackRF at 751 MHz, one LTE cell (PCI 25), -13.94 +- 0.07 ppm in 2 s; its half-frames alone
-14.6 +- 0.5 ppm.
"""
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

RADIO_PPM = 25.0       # A radio's own reference (TCXOs: 0.5-2.5 ppm; HackRF and RTL-SDR crystals 10-50)
DISCIPLINED_PPM = 0.05 # One locked to an external reference or a GPSDO
LNB_HZ = 1e6           # An LNB's LO error (a PLL LNB: 10-50 kHz; a DRO: a MHz or more)
LTE_RASTER = 100e3     # LTE carriers sit on a 100 kHz raster (EARFCN, 3GPP TS 36.101 5.7.3)

@dataclass
class FrequencyReference:
    """A carrier of known frequency

    measured: where the capture puts it [Hz over the air]; true: where it is; raster: instead of true, the step of the grid it sits on (resolved by `fit` when it can be); sigma: the
    measurement's standard error [Hz]; what: a name"""
    measured: float
    true: Optional[float] = None
    raster: Optional[float] = None
    sigma: float = 100.0
    what: str = ""

@dataclass
class RateReference:
    """A clock error measured against a known rate

    ppm (positive: the receiver's clock runs fast), sigma [ppm], what"""
    ppm: float
    sigma: float = 1.0
    what: str = ""

def rate(measured, nominal, sigma_ppm=1.0, what=""):
    """A RateReference from a measured rate and its known value

    A fast receiver clock reads a rate (a symbol rate, a frame rate) low"""
    return RateReference((nominal/measured - 1)*1e6, sigma_ppm, what)

def _geometry(context, rf):
    """(IF of a carrier at `rf` [Hz], its sign s: +1 through no LO or one below, -1 above, whether an LO is there)"""
    chain = getattr(context, "chain", None)
    if chain is None or not chain.lo:
        return rf, 1.0, False
    return chain.to_if(rf), (-1.0 if chain.invert else 1.0), True

@dataclass
class Calibration:
    """The receive chain's errors, and their uncertainty

    ppm (the radio's reference, positive fast) and lo_error [Hz] (an LNB's LO, positive high), their covariance `cov` (lo_error, ppm), the references they came from (`used`; `ambiguous`: rasters left
    unresolved, with the true frequencies they could be) and the context they apply to"""
    ppm: float = 0.0
    lo_error: float = 0.0
    cov: np.ndarray = field(default_factory=lambda: np.diag([0.0, 0.0]))
    used: list = field(default_factory=list)
    ambiguous: list = field(default_factory=list)
    context: object = None

    @property
    def ppm_sigma(self):
        return float(np.sqrt(self.cov[1, 1]))

    @property
    def lo_sigma(self):
        return float(np.sqrt(self.cov[0, 0]))

    def _row(self, rf):
        """d(measured - true)/d(lo_error, ppm) for a carrier at `rf`"""
        f_if, s, lo = _geometry(self.context, rf)
        return np.array([-1.0 if lo else 0.0, -s*f_if*1e-6])

    def frequency(self, measured):
        """(the true frequency of a carrier the capture puts at `measured` [Hz over the air], its standard error [Hz])"""
        g = self._row(measured)
        err = float(g @ [self.lo_error, self.ppm])
        return measured - err, float(np.sqrt(g @ self.cov @ g))

    def rate(self, measured):
        """(the true value of a rate measured with the receiver's clock (a symbol rate), its standard error)"""
        k = 1 + self.ppm*1e-6
        return measured*k, abs(measured)*self.ppm_sigma*1e-6

    def __str__(self):
        out = f"radio reference {self.ppm:+.3f} +- {self.ppm_sigma:.3f} ppm"
        if self.lo_sigma > 0 or self.lo_error:
            out += f", LO {self.lo_error/1e3:+.3f} +- {self.lo_sigma/1e3:.3f} kHz"
        out += f" ({len(self.used)} references" + (f", {len(self.ambiguous)} ambiguous" if self.ambiguous else "") + ")"
        return out

def priors(context):
    """The prior standard errors (lo_error [Hz], ppm) of a receive context"""
    clock = str(getattr(context, "clock_source", None) or "").lower()
    ppm = DISCIPLINED_PPM if clock in ("external", "gpsdo", "gps", "ext", "mimo") else RADIO_PPM
    chain = getattr(context, "chain", None)
    return (LNB_HZ if chain is not None and chain.lo else 0.0), ppm

def fit(references, context=None, prior=None):
    """The Calibration the references give

    For captures received through `context` (a pysdrlib RxContext; None: no chain, the radio's own reference). prior: (lo_error, ppm) standard errors instead of `priors(context)`"""
    s_lo, s_ppm = prior if prior is not None else priors(context)
    exact = [r for r in references if isinstance(r, RateReference) or r.true is not None]
    rasters = [r for r in references if isinstance(r, FrequencyReference) and r.true is None]
    resolved = []
    while True:
        cal = _solve(exact + resolved, context, s_lo, s_ppm)
        done = []
        for r in rasters:
            if r in resolved:
                continue
            centre, sd = cal.frequency(r.measured)
            if sd < r.raster/4: # The rest pins it: the raster point it is on
                resolved.append(FrequencyReference(r.measured, round(centre/r.raster)*r.raster, None, r.sigma,
                                                   f"{r.what} (on the {r.raster/1e3:g} kHz raster)"))
                done.append(r)
        if not done:
            break
        rasters = [r for r in rasters if r not in done]
    for r in rasters:
        centre, sd = cal.frequency(r.measured)
        k = np.arange(np.floor((centre - 2*sd)/r.raster), np.ceil((centre + 2*sd)/r.raster) + 1)
        cal.ambiguous.append((r, [float(v) for v in k*r.raster]))
    return cal

def _solve(refs, context, s_lo, s_ppm):
    rows, ys, ws = [], [], []
    probe = Calibration(context=context)
    for r in refs:
        if isinstance(r, RateReference):
            rows.append([0.0, 1.0]), ys.append(r.ppm), ws.append(1/r.sigma**2)
        else:
            rows.append(list(probe._row(r.true))), ys.append(r.measured - r.true), ws.append(1/r.sigma**2)
    for i, s in enumerate((s_lo, s_ppm)): # Priors: zero, within s (none, 0: held at zero)
        row = [0.0, 0.0]
        row[i] = 1.0
        rows.append(row), ys.append(0.0), ws.append(1/s**2 if s > 0 else 1e30)
    A, y, W = np.array(rows), np.array(ys), np.diag(ws)
    cov = np.linalg.inv(A.T @ W @ A)
    lo, ppm = cov @ A.T @ W @ y
    return Calibration(ppm=float(ppm), lo_error=float(lo), cov=cov, used=list(refs), context=context)

# ---- references from signals in a capture

def lte(x, Fs, offset=0.0, cf=0.0, seconds=0.4, span=60e3):
    """The references an LTE cell gives

    [FrequencyReference on the 100 kHz raster, RateReference of the sample clock], [] when no cell is found. x: samples at Fs (the capture's, any rate), the carrier at `offset` Hz from its centre `cf`
    (over the air, so the frequency reference is too). The PSS and SSS (the central 0.93 MHz) at 1.92 MS/s: the
    carrier is moved to the centre and resampled by an exact ratio, which keeps the capture's clock in its timing"""
    from math import gcd
    from rpps.base import Data
    from rpps.grid.lte import LteCell
    from rpps.signal.sample.re import Polyphase
    x = np.asarray(x[:int(seconds*Fs)])
    y = (x*np.exp(-2j*np.pi*offset/Fs*np.arange(len(x)))).astype(np.complex64)
    fs = 1.92e6
    if abs(Fs/fs - round(Fs/fs)) > 1e-9 or round(Fs/fs) != 1:
        up, down = int(round(fs)), int(round(Fs))
        g = gcd(up, down)
        rs = Polyphase(up//g, down//g)
        y = np.asarray(rs.run(y), dtype=np.complex64)
    d = Data(y, Fs=fs, flush=True)
    d + LteCell(span=span)
    e = d.stream.est.get("cell", {})
    if not e.get("locked") or e.get("half_frames", 0) < 8:
        return []
    what = f"LTE PCI {e['pci']}"
    halves = e["half_frames"]
    return [FrequencyReference(cf + offset + e["cfo"], raster=LTE_RASTER, sigma=50.0, what=what),
            RateReference(e["ppm"], sigma=max(0.3, 40/halves), what=f"{what} half-frames ({halves})")]

def extents(boxes, gap=100e3, min_snr=10.0):
    """The bands signals occupy, from survey boxes

    Boxes over `min_snr` dB merged where they are less than `gap` apart, as (low, high) [Hz from the centre]. An OFDM carrier's resource blocks and reference signals come out of a survey as many boxes;
    its occupied band is symmetric about its carrier, so the extent's middle is the carrier"""
    spans = sorted((s.mid - s.width/2, s.mid + s.width/2) for s in boxes if getattr(s, "snr", min_snr) >= min_snr)
    out = []
    for lo, hi in spans:
        if out and lo - out[-1][1] < gap:
            out[-1][1] = max(out[-1][1], hi)
        else:
            out.append([lo, hi])
    return [tuple(e) for e in out]

def find(x, Fs, cf=0.0, survey=None, widths=(1.0e6, 21e6)):
    """The references the signals in a capture give

    Every band `widths` wide (LTE's 1.4 to 20 MHz, `extents` of the survey's boxes) tried as an LTE cell at its middle. x: samples at Fs, centred on `cf` over the air; survey: the
    boxes (rpps.analysis.spectrum.find) when made already"""
    from rpps.analysis.spectrum import find as _find
    boxes = survey if survey is not None else _find.find(x, Fs, cf)
    out = []
    for lo, hi in extents(boxes):
        if widths[0] <= hi - lo <= widths[1]: # (its edges are ragged: searched 3% of its width either side)
            out += lte(x, Fs, (lo + hi)/2, cf, span=max(60e3, 0.03*(hi - lo)))
    return out

def calibrate(source, seconds=0.4, references=(), prior=None):
    """The Calibration of a source's receive chain

    From the references in its first `seconds` (`find`) and those given (a beacon's FrequencyReference, a RateReference). The source's samples are read with `peek`, so
    they are still there for what comes after (a Device's are not: capture it into Memory first)"""
    x = source.peek(int(seconds*source.Fs))
    refs = find(x, source.Fs, source.cf) + list(references)
    return fit(refs, getattr(source, "context", None), prior)


def main():
    import argparse
    import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); import _paths  # noqa: E401,E402,F401
    import reference_io
    ap = argparse.ArgumentParser(description="Measure the radio's reference error (and an LNB's LO) from LTE cells in a capture")
    reference_io.add_source_args(ap)
    a = ap.parse_args()
    src = reference_io.source(a)
    cal = calibrate(src, seconds=a.seconds)
    for r in cal.used:
        print(f"  {r.what}: " + (f"{r.ppm:+.3f} +- {r.sigma:.3f} ppm" if isinstance(r, RateReference) else
                                f"seen at {r.measured/1e6:.6f} MHz, is at {r.true/1e6:.6f} MHz"))
    for r, could in cal.ambiguous:
        print(f"  {r.what}: on the {r.raster/1e3:g} kHz raster, but could be any of {', '.join(f'{v/1e6:.4f}' for v in could)} MHz")
    if not cal.used:
        print("no references found (an LTE cell 1.4 to 20 MHz wide in the capture)")
        return 1
    print(cal)
    ctx = getattr(src, "context", None)
    total = (getattr(ctx, "ppm", 0.0) or 0.0) + cal.ppm # (what it measured is on top of the correction already applied)
    print(f"pysdrlib: sdr.set_freq_correction({total:+.3f}, accuracy={cal.ppm_sigma:.3f})")
    if ctx is not None and ctx.chain is not None and ctx.chain.lo:
        print(f"frontends chain: lo = {ctx.chain.lo + cal.lo_error:.0f}, lo_accuracy = {max(cal.lo_sigma, 1.0):.0f}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
