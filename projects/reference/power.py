"""Calibrate a receiver's level: what dBFS is in dBm at its antenna port, at the gains and frequency it is set to

    python projects/reference/power.py floor --nf 8 --freq 751 --fs 10           # no equipment: the input terminated
    python projects/reference/power.py tone --dbm -60 --freq 751 --fs 10         # a signal generator's tone, at the centre
    python projects/reference/power.py yfactor --enr 15 --on on.sigmf --off off.sigmf --fs 10 --freq 751   # a noise source

Prints the offset (dBm at the port = dBFS + offset) and the pysdrlib call that keeps it,
`set_level_calibration(offset, accuracy=)`. RPPS reads it back from the receive context (`RxContext.level_offset`,
`level_accuracy`): the signal profile's `level` step then gives each carrier in dBm. It holds only for the gains and
frequency it was measured at.

Three ways, by what there is to measure against:

    floor     the radio's own noise, with nothing on its input (a 50 ohm terminator, or the antenna off): the floor is
              kT0 = -174 dBm/Hz raised by the noise figure, so the offset is -174 + NF less the floor in dBFS/Hz. The NF
              comes from the datasheet (a few dB off, which is the accuracy), and the gain must be high enough that the
              analog noise, not the ADC's, sets the floor: 6 dB more gain should lift the floor by about 6 dB
    tone      a known level at the port (a signal generator through a known cable): its power in dBFS against its dBm
    yfactor   a noise source of known ENR, on and off: Y = on/off gives the noise figure (F = ENR/(Y - 1)), then as floor
"""
import math
from dataclasses import dataclass

import numpy as np

KT0 = -174.0  # dBm/Hz: thermal noise at 290 K


@dataclass
class Level:
    """offset: dBm at the port = dBFS + offset. accuracy: its standard error [dB]. how: what it came from. nf: the noise
    figure [dB] it used or measured (None for a tone)"""
    offset: float
    accuracy: float
    how: str
    nf: float = None

    def __str__(self):
        nf = f", noise figure {self.nf:.1f} dB" if self.nf is not None else ""
        return f"{self.how}: dBm at the port = dBFS {self.offset:+.1f} dB +- {self.accuracy:.1f} dB{nf}"


def floor(floor_dbfs_hz, nf_db, nf_sigma=2.0, measured_sigma=0.2):
    """The Level from the noise floor with nothing on the input [dBFS/Hz] and the receiver's noise figure (a datasheet's,
    known to `nf_sigma` dB)"""
    return Level(KT0 + nf_db - floor_dbfs_hz, math.hypot(nf_sigma, measured_sigma),
                 f"the floor {floor_dbfs_hz:.1f} dBFS/Hz with a {nf_db:g} dB noise figure", nf_db)


def tone(measured_dbfs, known_dbm, known_sigma=0.5, measured_sigma=0.1):
    """The Level from a tone of known power at the port (`known_sigma`: the generator's and the cable's accuracy)"""
    return Level(known_dbm - measured_dbfs, math.hypot(known_sigma, measured_sigma),
                 f"a {known_dbm:g} dBm tone read at {measured_dbfs:.1f} dBFS")


def y_factor(on_dbfs_hz, off_dbfs_hz, enr_db, enr_sigma=0.2, measured_sigma=0.1):
    """The Level from a noise source of excess noise ratio `enr_db`: the floor with it on and off [dBFS/Hz]"""
    y = 10**((on_dbfs_hz - off_dbfs_hz)/10)
    if y <= 1.0:
        raise ValueError(f"the noise source raised the floor by {on_dbfs_hz - off_dbfs_hz:.2f} dB: it isn't reaching the input")
    nf = 10*math.log10(10**(enr_db/10)/(y - 1))
    # (a dB of ENR is a dB of NF; the Y measurement's error grows as Y nears 1)
    sigma = math.hypot(enr_sigma, measured_sigma*y/(y - 1))
    lv = floor(off_dbfs_hz, nf, sigma, 0.0)
    lv.how = f"a {enr_db:g} dB ENR noise source (Y {10*math.log10(y):.2f} dB)"
    return lv


def floor_dbfs_hz(x, Fs, nfft=4096, middle=0.5):
    """The noise floor [dBFS/Hz] of samples (|x| = 1 full scale) over the `middle` share of the band (past the
    radio's filter roll-off), from rp.analysis's floor fit, which leaves out what stands over it"""
    from rpps.analysis.spectrum.find import fit_floor, spectrogram
    P, hop = spectrogram(np.asarray(x), Fs, nfft, overlap=0.5)
    fl = fit_floor(P, hop).floor
    a, b = int(nfft*(1 - middle)/2), int(nfft*(1 + middle)/2)
    return float(10*np.log10(np.median(fl[a:b])))


def tone_dbfs(x, Fs, offset=0.0, span=None):
    """The power [dBFS] of the strongest signal within `span` Hz of `offset` (default: a tenth of the band), the band's
    noise taken out (rp.analysis.survey)"""
    import rpps as rp
    sv = rp.analysis.survey(np.asarray(x), Fs=Fs, more=0, baud=False)
    span = Fs/10 if span is None else span
    near = [s for s in sv if abs(s.peak_freq - offset) <= span and s.carrier is not None]
    if not near:
        raise ValueError(f"no tone within {span/1e3:g} kHz of {offset/1e3:+g} kHz")
    return max(s.carrier for s in near)


def _report(lv, src):
    ctx = getattr(src, "context", None)
    print(lv)
    if ctx is not None and ctx.gains:
        at = ", ".join(f"{k} {v:g} dB" for k, v in ctx.gains.items() if v is not None)
        print(f"  at {at}, {ctx.freq/1e6:.3f} MHz: it holds only there")
    print(f"pysdrlib: sdr.set_level_calibration({lv.offset:+.2f}, accuracy={lv.accuracy:.2f})")


def main():
    import argparse
    import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); import _paths  # noqa: E401,E402,F401
    import reference_io
    ap = argparse.ArgumentParser(description="Calibrate a receiver's level: dBFS to dBm at its antenna port")
    sub = ap.add_subparsers(dest="how", required=True)
    p_fl = sub.add_parser("floor", help="the radio's own noise, nothing on its input, and its noise figure")
    p_fl.add_argument("--nf", type=float, required=True, help="the receiver's noise figure [dB] at these gains (datasheet)")
    p_fl.add_argument("--nf-sigma", type=float, default=2.0, help="how well the noise figure is known [dB] (default 2)")
    p_tn = sub.add_parser("tone", help="a known tone at the port")
    p_tn.add_argument("--dbm", type=float, required=True, help="the tone's power at the port [dBm]")
    p_tn.add_argument("--sigma", type=float, default=0.5, help="how well that is known [dB] (default 0.5)")
    p_tn.add_argument("--offset", type=float, default=0.0, help="the tone's offset from the centre [kHz]")
    p_yf = sub.add_parser("yfactor", help="a noise source, on and off")
    p_yf.add_argument("--enr", type=float, required=True, help="the noise source's ENR at this frequency [dB]")
    p_yf.add_argument("--on", required=True, help="the capture with the source on")
    p_yf.add_argument("--off", required=True, help="the capture with it off")
    for p in (p_fl, p_tn, p_yf):
        reference_io.add_source_args(p, fs=10.0, seconds=0.2, freq_required=p is not p_yf)
    a = ap.parse_args()
    if a.how == "yfactor":
        on, off = reference_io.source(a, a.on), reference_io.source(a, a.off)
        n = int(a.seconds*on.Fs)
        lv = y_factor(floor_dbfs_hz(on.peek(n), on.Fs), floor_dbfs_hz(off.peek(n), off.Fs), a.enr)
        _report(lv, off)
        return 0
    src = reference_io.source(a)
    x = src.peek(int(a.seconds*src.Fs))
    if a.how == "floor":
        lv = floor(floor_dbfs_hz(x, src.Fs), a.nf, a.nf_sigma)
    else:
        lv = tone(tone_dbfs(x, src.Fs, a.offset*1e3), a.dbm, a.sigma)
    _report(lv, src)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
