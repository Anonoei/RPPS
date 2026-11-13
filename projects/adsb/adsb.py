"""ADS-B / Mode S receiver for 1090 MHz: aircraft identification, altitude, velocity and position, with a gain search that counts valid messages

    python projects/adsb/adsb.py --tune                           # try gains, keep the one with the most CRC-valid messages
    python projects/adsb/adsb.py --seconds 30 --gain 0.67
    python projects/adsb/adsb.py data/adsb/a.cf64                 # a saved 2 MS/s capture (cf64 at 1090 MHz)

Mode S replies are 8 us of preamble (pulses at 0, 1, 3.5, 4.5 us) and 56 or 112 bits of pulse-position modulation at 1 Mbit/s: a pulse in the first half of a microsecond is a 1, in the second half a 0.
At 2 MS/s that is two samples per bit. A message counts when its 24-bit CRC (poly 0xFFF409, rpps.link.crc) is zero (DF17 / DF18 extended squitter, DF11 all-call replies carry the interrogator id in the CRC so
only DF17 / 18 are kept). The ICAO address is the aircraft; type codes 1-4 give the callsign, 9-18 barometric altitude and a CPR position (decoded globally from an even and an odd frame within 10 s),
19 ground speed and track. ADS-B is a public broadcast meant to be received by anyone. Meant to be upstreamed into RPPS.
"""
import argparse
import os
import sys
import time
from collections import defaultdict

import numpy as np

import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); import _paths  # noqa: E402,F401
import radio_io
import rpps as rp

FS = 2e6
_CRC = rp.link.load("crc", "crc24_mode_s")._enc
CHARS = "#ABCDEFGHIJKLMNOPQRSTUVWXYZ##### ###############0123456789######"


def messages(x):
    """Yield (sample index, 14 bytes) of every CRC-valid long Mode S message in a 2 MS/s complex capture"""
    m = np.abs(x).astype(np.float32)
    n = len(m) - 16*2 - 112*2
    pre = m[0:n] + m[2:n + 2] + m[7:n + 7] + m[9:n + 9] # pulses at 0, 1, 3.5, 4.5 us
    quiet = m[1:n + 1] + m[3:n + 3] + m[4:n + 4] + m[5:n + 5] + m[6:n + 6] + m[8:n + 8]
    cand = np.where((pre > 1.5*quiet*4/6*1.0 + 1e-9) & (pre > 4*np.median(m)))[0]
    last = -1000
    for i in cand:
        if i - last < 240: # inside the previous message
            continue
        s = m[i + 16:i + 16 + 224]
        if len(s) < 224:
            break
        bits = (s[0::2] > s[1::2]).astype(np.uint8)
        by = np.packbits(bits).tobytes()
        df = by[0] >> 3
        if df in (17, 18) and _CRC.compute(by[:11]) == int.from_bytes(by[11:14], "big"):
            last = i
            yield int(i), by


def cpr_global(e, o, t_even_newer):
    """(lat, lon) from an even and an odd airborne CPR frame (lat, lon in 17-bit units), or None"""
    la0, lo0 = e; la1, lo1 = o
    j = int(np.floor(59*la0/131072 - 60*la1/131072 + 0.5))
    NL = lambda lat: 1 if abs(lat) >= 87 else int(np.floor(2*np.pi/np.arccos(1 - (1 - np.cos(np.pi/30))/np.cos(np.pi*abs(lat)/180)**2)))
    r0 = 360/60*((j % 60) + la0/131072); r1 = 360/59*((j % 59) + la1/131072)
    r0 = r0 - 360 if r0 >= 270 else r0; r1 = r1 - 360 if r1 >= 270 else r1
    if NL(r0) != NL(r1):
        return None
    lat = r0 if t_even_newer else r1
    nl = NL(lat); ni = max(nl - (0 if t_even_newer else 1), 1)
    mm = int(np.floor(lo0*(nl - 1)/131072 - lo1*nl/131072 + 0.5))
    lon = 360/ni*((mm % ni) + (lo0 if t_even_newer else lo1)/131072)
    lon = lon - 360 if lon >= 180 else lon
    return lat, lon


class Aircraft:
    def __init__(self):
        self.call = None; self.alt = None; self.speed = None; self.track = None; self.pos = None; self.n = 0; self.cpr = {}

    def update(self, by, t):
        self.n += 1; me = int.from_bytes(by[4:11], "big"); tc = me >> 51
        if 1 <= tc <= 4:
            self.call = "".join(CHARS[(me >> (42 - 6*k)) & 63] for k in range(8)).strip()
        elif 9 <= tc <= 18 or 20 <= tc <= 22:
            ac = (me >> 36) & 0xFFF; q = (ac >> 4) & 1 # 12-bit altitude after TC (5), SS (2), NIC (1); bit 4 is the Q bit
            if tc <= 18 and q:
                n = ((ac & 0xFE0) >> 1) | (ac & 0xF); self.alt = n*25 - 1000
            odd = (me >> 34) & 1; la = (me >> 17) & 0x1FFFF; lo = me & 0x1FFFF
            self.cpr[odd] = (la, lo, t)
            if 0 in self.cpr and 1 in self.cpr and abs(self.cpr[0][2] - self.cpr[1][2]) < 10:
                p = cpr_global(self.cpr[0][:2], self.cpr[1][:2], self.cpr[0][2] > self.cpr[1][2])
                if p and -90 <= p[0] <= 90:
                    self.pos = p
        elif tc == 19 and ((me >> 48) & 7) in (1, 2):
            ew = ((me >> 32) & 0x3FF) - 1; ns = ((me >> 21) & 0x3FF) - 1
            if ((me >> 42) & 1): ew = -ew
            if ((me >> 31) & 1): ns = -ns
            self.speed = float(np.hypot(ew, ns)); self.track = float(np.degrees(np.arctan2(ew, ns)) % 360)


def decode(x):
    ac = defaultdict(Aircraft); n = 0
    for i, by in messages(x):
        icao = by[1:4].hex(); ac[icao].update(by, i/FS); n += 1
    return n, ac


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file", nargs="?", help="saved cf64 capture at 2 MS/s, 1090 MHz")
    ap.add_argument("--seconds", type=float, default=30.0); ap.add_argument("--tune", action="store_true", help="search the gain for the most valid messages")
    radio_io.add_gain_arg(ap, 0.67)
    ap.add_argument("--keep", help="save the live capture (cf64)")
    a = ap.parse_args(argv)
    if a.file:
        x = np.fromfile(a.file, np.complex64)
    else:
        sdr = radio_io.open_radio(cf=1090e6, fs=FS); sdr.start_rx(); time.sleep(0.2)
        try:
            if a.tune:
                best = None
                for g in radio_io.GAINS_FINE:
                    radio_io.set_gain(sdr, g); time.sleep(0.05)
                    x = radio_io.grab(sdr, int(FS*6)); n, _ = decode(x.astype(np.complex64)); c = radio_io.clipped(x)
                    print(f"gain {radio_io.describe(sdr)}: {n:3d} valid messages in 6 s, clipped {c:.0e}", flush=True)
                    if c < 1e-3 and (best is None or n > best[0]):
                        best = (n, g)
                print(f"best: gain {best[1]:.2f} ({best[0]} messages)")
                a.gain = best[1]
            radio_io.set_gain(sdr, a.gain); time.sleep(0.1)
            x = radio_io.grab(sdr, int(FS*a.seconds))
        finally:
            sdr.stop_rx(); sdr.close()
        x = x.astype(np.complex64)
        if a.keep:
            x.tofile(a.keep)
    n, ac = decode(x)
    print(f"{n} valid extended-squitter messages from {len(ac)} aircraft in {len(x)/FS:.0f} s")
    for icao, p in sorted(ac.items(), key=lambda kv: -kv[1].n):
        print(f"  {icao}  {p.call or '-':8s} msgs {p.n:3d}  alt {p.alt if p.alt is not None else '-':>6} ft  speed {('%.0f kt' % p.speed) if p.speed else '-':>7}  track {('%.0f' % p.track) if p.track is not None else '-':>4}  "
              + (f"pos {p.pos[0]:.3f}, {p.pos[1]:.3f}" if p.pos else ""))
    return ac


if __name__ == "__main__":
    main()
