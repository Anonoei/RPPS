"""Decode one DVB-S carrier of a capture to MPEG-TS and print its service information: the network and where it is sent
(NIT: frequency, orbital position, polarization), the services (SDT). Names the satellite a capture was pointed at.

Usage: python tests/dev/dvb_si.py capture.cs16 FREQ_HZ [--fs 100e6] [--seconds 3] [--ts out.ts]
    FREQ_HZ: the carrier, from the capture's centre (as the survey prints it); the sample rate comes from a gqrx name
rp.sync.AcquireFEC finds the code; once it has, its profile (dvbs/qpsk_3_4, ...) decodes the symbols as they arrive.
"""
import argparse
import os
import re
import sys

import numpy as np
import rpps as rp

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "projects", "atsc"))

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("path")
    ap.add_argument("freq", type=float)
    ap.add_argument("--fs", type=float, default=None)
    ap.add_argument("--seconds", type=float, default=3.0)
    ap.add_argument("--ts", default=None)
    a = ap.parse_args()
    m = re.search(r"_(\d+)\.cs16$", a.path)
    Fs = a.fs or float(m.group(1))
    src = rp.source.File("ci32t", a.path, Fs=Fs)
    block = 16384
    rx, dec, bits = None, None, []
    for i, blk in enumerate(src(block)):
        if i*block >= a.seconds*Fs:
            break
        if rx is None:
            sig = rp.analysis.survey(src).pick(freq=a.freq)
            print(f"carrier: {sig.describe()}")
            rx = sig.channel().front + rp.sync.Acquire() + rp.sync.time.Mat() + rp.sync.phase.FFCarrier() + rp.sync.AcquireFEC()
        out = blk + rx
        fec = out.stream.est.get("fec")
        if dec is None and fec is not None and fec.found and fec.complete:
            print(f"code: {fec}")
            if not fec.profile_name:
                sys.exit(f"the code isn't a shipped transport-stream profile: {fec}")
            dec = fec.rx()
        if dec is not None and len(out):
            bits.append(np.asarray(dec(out).data).astype(bool))
    if not bits:
        sys.exit(f"nothing decoded: {out.stream.est.get('fec')}")
    b = np.packbits(np.concatenate(bits))
    ts = b[:len(b)//188*188]
    print(f"{len(ts)//188} TS packets, sync bytes ok {np.mean(ts.reshape(-1, 188)[:, 0] == 0x47):.3f}")
    if a.ts:
        ts.tofile(a.ts)
    import ts_info
    ts_info.dvb_si(ts.tobytes())

if __name__ == "__main__":
    main()
