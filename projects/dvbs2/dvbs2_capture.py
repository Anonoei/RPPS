"""Decode the DVB-S2 carriers in a wideband capture with the dvbs2/normal profile alone

Usage: python projects/dvbs2/dvbs2_capture.py [file.cs16 [sample rate]] [blocks] [first signal] [last signal]

Each carrier the survey finds is shifted to baseband and decimated (rp.signal.channelize), resampled to 4 samples per
symbol, and handed to the profile from raw samples: its own matched filter, timing (Oerder & Meyr) and carrier
(8th power) estimators, frame sync, PLS and the rest. The symbol rate and roll-off are what a channel plan would give;
here they come from rpps.analysis.modulation.detect on the first stretch of the carrier. The carrier's frequency offset
and the clock error are left in: the profile finds them.
"""
import re
import sys
from fractions import Fraction

import numpy as np
import rpps as rp

FILE = "data/610BE338EA883D94_gqrx_20230303_030038_3690035999_100000000.cs16"

def main():
    args = sys.argv[1:]
    path = args.pop(0) if args and args[0].endswith(".cs16") else FILE
    Fs = float(re.search(r"_(\d+)(?:\.cs16|/chunk)", path + "/chunk").group(1))
    if args and float(args[0]) > 1e5:
        Fs = float(args.pop(0))
    blocks = int(args[0]) if len(args) > 0 else 450
    first = int(args[1]) if len(args) > 1 else 0
    last = int(args[2]) if len(args) > 2 else 99
    sigs = rp.analysis.survey(rp.source.File("ci32t", path, Fs=Fs))
    print(f"{len(sigs)} signals", flush=True)
    for j, s in enumerate(sigs):
        if not first <= j <= last:
            continue
        ch = s.channel()
        src = rp.source.File("ci32t", path, Fs=Fs)
        x = np.concatenate([np.asarray((rp.Data(b.data, Fs=Fs, stream=st) + ch.front).data)
                            for st in [rp.base.Stream(Fs)] for i, b in zip(range(blocks), src(1024*16))])
        det = rp.analysis.modulation.detect(x[:2**16], ch.Fs)
        if det is None or det.mod not in ("qpsk", "8psk"):
            print(f"signal {j}: {s.mid/1e6:+.3f} MHz: not QPSK/8PSK ({det and det.mod})", flush=True)
            continue
        r = Fraction(4*det.baud/ch.Fs).limit_denominator(400) # to 4 samples per symbol, near enough: the timing tracks the rest
        y = np.asarray((rp.Data(x, Fs=ch.Fs, flush=True) + rp.signal.sample.re.Polyphase(r.numerator, r.denominator)).data)
        y = (y/np.sqrt(np.mean(np.abs(y)**2))).astype(np.complex64)
        beta = min((0.2, 0.25, 0.35), key=lambda b: abs(b - det.beta))
        p = rp.profile.load("dvbs2/normal", pulse=f"rrc/0p{int(round(beta*100)):02d}")
        out = p.rx()(rp.Data(y, Fs=ch.Fs*r.numerator/r.denominator, sps=4, flush=True))
        est, ctl = out.stream.est, out.stream.control
        heads = [f for _, f, _ in ctl.fields("pls")]
        kinds = sorted({(f["modcod"], f["pilots"]) for f in heads})
        bb = est.get("dvbs2_bb", {"frames": 0, "crc_failed": 0})
        tm, cr = est.get("timing", {}), est.get("carrier", {})
        print(f"signal {j}: {s.mid/1e6:+.3f} MHz, {det.baud/1e6:.4f} MBd, roll-off {beta}: timing {tm.get('ppm', 0) or 0:+.0f} ppm, "
              f"carrier {(cr.get('frequency') or 0)*det.baud/1e3:+.1f} kHz; {len(heads)} PLHEADERs {kinds}, "
              f"{bb['frames']} BBHEADERs with a good CRC, {bb['crc_failed']} bad, {len(out)} data bits", flush=True)

if __name__ == "__main__":
    main()
