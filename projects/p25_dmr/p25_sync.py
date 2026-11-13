"""Look for APCO P25 phase 1 (C4FM, 4800 baud, 12.5 kHz) frame sync in a capture, and read the network ID after each one

    python projects/p25_dmr/p25_sync.py data/fed/fed_407.cf64 --fs 4 --cf 407 --freq 407.6187 406.7437 406.4438 406.1562 --seconds 10

Per channel: mix to zero (the channel's strongest line is used as the carrier), 7 kHz low-pass, 48 kS/s (10 samples per symbol), FM demodulate, pick the best of 10 sample phases, scale to
levels +-1/+-3 and correlate with the 24-symbol frame sync 0x5575F5FF77FF (dibits 01 = +3, 00 = +1, 10 = -1, 11 = -3). Peaks above --corr are sync hits (a random signal never reaches 0.9; noise-free
sync gives 1.00). After each hit the 32 NID symbols are sliced and printed as NAC (12 bits) and DUID (4 bits).
The NID is BCH(63,16, t=11) over GF(2^6) (poly 0x43) plus an overall parity bit: it is decoded with rpps' `bchgf`, and a NAC / DUID counts only when the decode succeeds. The levels of each sync word
(all +-3) calibrate the slicer for the 32 NID symbols that follow. Limits: the fixed sample-phase pick is the only timing recovery. DMR and NXDN are also 4800 baud 4-level but use other sync words.
Meant to be upstreamed into RPPS.
"""
import argparse
from fractions import Fraction
import os
import sys
from collections import Counter

import numpy as np
from rpps.code.types.bchgf import bchgf
import rpps as rp
from rpps.logger import Logger
from rpps.signal.sample import resample_poly
from rpps.signal.filters import low_pass

import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); import _paths  # noqa: E402,F401
import burst_tones

DUID = {0x0: "HDU", 0x3: "TDU", 0x5: "LDU1", 0x7: "TSDU", 0xA: "LDU2", 0xC: "PDU", 0xF: "TDULC"}
_LEV = {1: 3.0, 0: 1.0, 2: -1.0, 3: -3.0} # dibit -> symbol level
SYNC = np.array([_LEV[int(format(0x5575F5FF77FF, "048b")[i:i + 2], 2)] for i in range(0, 48, 2)])


def symbols(x, fs, cf, freq):
    """Symbol-rate levels (nominally +-1 / +-3) of the channel at freq [Hz]"""
    z, fs1 = burst_tones.channel(x, fs, cf, freq)
    n = 2**14; k = len(z)//n # carrier: the strongest line of the spectrum averaged over the whole capture (the channel may be idle at the start)
    S = (np.abs(np.fft.fftshift(np.fft.fft(z[:k*n].reshape(k, n)*np.hanning(n), axis=1), axes=1))**2).mean(0); ff = np.fft.fftshift(np.fft.fftfreq(n, 1/fs1))
    c = ff[np.argmax(np.convolve(S, np.ones(8), "same"))]
    z = z*np.exp(-2j*np.pi*c*np.arange(len(z))/fs1)
    z = low_pass(121, 7e3, fs=fs1).run(z)
    q = Fraction(48000).limit_denominator(1) / Fraction(int(round(fs1))) # channel rate -> 48 kS/s (10 samples per symbol), whatever the capture's sample rate
    z = resample_poly(z, q.numerator, q.denominator)
    fi = np.convolve(np.diff(np.unwrap(np.angle(z)))*48000/(2*np.pi), np.ones(5)/5)[:len(z) - 1]; fi -= np.median(fi)
    ph = max(range(10), key=lambda p: np.mean(np.abs(fi[p::10]))); s = fi[ph::10]
    a = np.abs(s)
    k = np.mean(a[a > np.percentile(a, 50)])/2.0
    symbols.stream = (fi/k, ph) # the 48 kS/s discriminator, for per-frame timing (refine)
    return s/k, c


def refine(stream, i, n=420, span=6.0, step=0.5, sync=None, before=0):
    """Symbols of the frame whose sync was found at symbol i: re-pick the sampling instant of this frame (+-span samples, interpolated) as the one the sync matches best"""
    S_ = SYNC if sync is None else sync
    fi, ph = stream; s0 = i*10 + ph; best = None
    for off in np.arange(-span, span + 1e-9, step):
        t = s0 + off + 10*np.arange(len(S_))
        if t[0] < 0 or t[-1] >= len(fi):
            continue
        v = np.interp(t, np.arange(len(fi)), fi); c = np.dot(v, S_)/np.sqrt(np.dot(v, v)*np.dot(S_, S_) + 1e-12)
        if best is None or c > best[0]:
            best = (c, off)
    if best is None:
        return np.zeros(0)
    t = s0 + best[1] + 10*np.arange(-before, n)
    t = t[(t >= 0) & (t < len(fi))]
    return np.interp(t, np.arange(len(fi)), fi)


def sync_hits(sn, corr=0.9, sync=None):
    SYNC_ = SYNC if sync is None else sync; m = len(SYNC_)
    r = np.correlate(sn, SYNC_, "valid")/np.sqrt((SYNC_**2).sum()*np.maximum(1e-9, np.convolve(sn**2, np.ones(m), "valid")))
    return [i for i in range(1, len(r) - 1) if r[i] > corr and r[i] >= r[i - 1] and r[i] >= r[i + 1]], r


_BCH = bchgf(Logger("p25"), 63, 16, m=6, poly=0x43, t=11)


def slice_dibits(seg):
    """Symbol levels (+3 +1 -1 -3) -> dibits (01 00 10 11)"""
    return np.where(seg > 2, 1, np.where(seg > 0, 0, np.where(seg > -2, 2, 3)))


def nid(sn, i):
    """(NAC, DUID, corrected bit errors) of the NID after the sync at symbol i, or None when the BCH cannot correct it"""
    a = sn[i:i + len(SYNC)]; seg = sn[i + len(SYNC):i + len(SYNC) + 33]
    if len(seg) < 33:
        return None
    seg = np.delete(seg, 35 - len(SYNC))/(np.mean(a*SYNC)/9.0) # a status symbol follows every 35 symbols of the frame; the sync's +-3 levels set the scale
    bits = np.array([int(b) for b in "".join(format(int(v), "02b") for v in slice_dibits(seg))], np.uint8)
    word = _BCH.decode(bits[:63])
    if _BCH.errors[0] < 0:
        return None
    w = "".join(map(str, word.astype(int)))
    return int(w[:12], 2), int(w[12:16], 2), int(_BCH.errors[0])


# --- trunking signalling blocks (TSDU): 196 bits = 98 dibits, interleaved, 1/2-rate trellis coded -> 12 bytes (10 data + CRC-16). Layout checked against OP25's p25p1_fdma.cc: status dibits dropped
# (every 36th), blocks start 112 bits after the sync; the interleave takes dibit pairs from four columns that start at dibits 0, 26, 50 and 74.
_DEINTERLEAVE = rp.code.load("interleave", "p25_tsbk")._enc # 98 dibits from four columns (interleave/block/p25_tsbk)
_TRELLIS = rp.code.load("block", "trellis/p25_1_2")._dec # rate 1/2 trellis code, Viterbi decoded
_CRC = rp.link.load("crc", "crc16_gsm")._enc # CRC-CCITT, inverted remainder


def frame_bits(sn, i, nblocks=3):
    """Bits of the frame whose sync starts at symbol i with the status symbols removed, from the NID on (after calibrating the levels on the sync)"""
    a = sn[i:i + len(SYNC)]; g = np.mean(a*SYNC)/9.0
    n = 32 + 98*nblocks + 24
    seg = sn[i + len(SYNC):i + len(SYNC) + n + 12]/g
    keep = np.array([(len(SYNC) + k + 1) % 36 != 0 for k in range(len(seg))])
    d = slice_dibits(seg[keep])
    return np.array([int(b) for v in d for b in format(int(v), "02b")], np.uint8)


def tsbk_blocks(bits, nblocks=3):
    """Decode up to nblocks TSBKs from the bits after the NID (the NID's 64 bits are skipped): list of (12 bytes, crc ok)"""
    out = []
    for k in range(nblocks):
        blk = bits[64 + 196*k:64 + 196*(k + 1)]
        if len(blk) < 196:
            break
        bits96 = _TRELLIS.decode(_DEINTERLEAVE.decode(blk))
        res = [(int(bits96[2*d]) << 1) | int(bits96[2*d + 1]) for d in range(48)]
        by = bytes(sum(res[4*b + q] << (6 - 2*q) for q in range(4)) for b in range(12))
        out.append((by, _CRC.compute(by[:10]) == int.from_bytes(by[10:12], "big")))
    return out


class Tsbk:
    """Collects the standard (MFID 0) trunking broadcasts of a control channel: channel tables, system / site identity, adjacent sites, voice grants. Layouts as OP25's trunking.py"""
    def __init__(self):
        self.iden = {}; self.sys = {}; self.rfss = set(); self.adj = set(); self.grants = Counter(); self.opts = {}; self.other = Counter()

    SLOTS = [1, 1, 1, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2] # slots per carrier by the 4-bit channel type of IDEN_UP_TDMA (types 0-2 are FDMA / phase 1, 3 is the phase 2 two-slot carrier)

    def freq(self, ch):
        t = self.iden.get((ch >> 12) & 0xf)
        return None if t is None else (t[0] + t[1]*((ch & 0xfff)//t[3]))/1e6

    def slot(self, ch):
        t = self.iden.get((ch >> 12) & 0xf)
        return (ch & 1) if t is not None and t[3] > 1 else None

    def add(self, by):
        if by[1] != 0:
            self.other[f"MFID {by[1]:#x} op {by[0] & 0x3f:#x}"] += 1; return
        op = by[0] & 0x3f; v = int.from_bytes(by[:10], "big") << 16
        if op in (0x3D, 0x34):
            i = (v >> 76) & 0xf; spac = (v >> 48) & 0x3ff; base = ((v >> 16) & 0xffffffff)*5
            if i not in self.iden or self.iden[i][3] == 1:
                self.iden[i] = (base, spac*125, "VU" if op == 0x34 else "", 1)
        elif op == 0x33:
            i = (v >> 76) & 0xf; ctype = (v >> 72) & 0xf; spac = (v >> 48) & 0x3ff; base = ((v >> 16) & 0xffffffff)*5
            self.iden[i] = (base, spac*125, f"TDMA type {ctype}", self.SLOTS[ctype])
        elif op == 0x3B:
            self.sys["wacn"] = (v >> 52) & 0xfffff; self.sys["sysid"] = (v >> 40) & 0xfff; self.sys["ch"] = (v >> 24) & 0xffff
        elif op == 0x3A:
            self.rfss.add(((v >> 56) & 0xfff, (v >> 48) & 0xff, (v >> 40) & 0xff, (v >> 24) & 0xffff))
        elif op == 0x3C:
            self.adj.add(((v >> 56) & 0xfff, (v >> 48) & 0xff, (v >> 40) & 0xff, (v >> 24) & 0xffff))
        elif op == 0x00: # group voice channel grant: service options, channel, group, source unit
            self.grants[((v >> 56) & 0xffff, (v >> 40) & 0xffff, (v >> 16) & 0xffffff)] += 1
            self.opts.setdefault((v >> 40) & 0xffff, Counter())[(v >> 72) & 0xff] += 1 # service options: 0x80 emergency, 0x40 encrypted (protected), 0x20 duplex, 0x10 packet mode, 0x07 priority
        elif op == 0x02: # group voice channel grant update: two (channel, group) pairs, no source unit (layout as I remember TIA-102.AABC / OP25; not checked against a copy of the standard)
            self.grants[((v >> 64) & 0xffff, (v >> 48) & 0xffff, 0)] += 1; self.grants[((v >> 32) & 0xffff, (v >> 16) & 0xffff, 0)] += 1
        else:
            self.other[f"op {op:#x}"] += 1

    def report(self):
        for i in sorted(self.iden):
            b, st, k, sl = self.iden[i]; print(f"  channel table {i}: base {b/1e6:.4f} MHz, step {st/1e3:g} kHz {k}" + (f"  <- {sl} slots per carrier (TDMA)" if sl > 1 else ""))
        if self.sys:
            f = self.freq(self.sys["ch"]); print(f"  network: WACN {self.sys['wacn']:#x}, system ID {self.sys['sysid']:#x}" + (f", control channel {f:.4f} MHz" if f else ""))
        for sy, rf, si, ch in sorted(self.rfss):
            f = self.freq(ch); print(f"  this site: system {sy:#x}, RFSS {rf}, site {si}" + (f", control channel {f:.4f} MHz" if f else f", channel {ch:#x}"))
        for sy, rf, si, ch in sorted(self.adj):
            f = self.freq(ch); print(f"  adjacent site: system {sy:#x}, RFSS {rf}, site {si}" + (f", control channel {f:.4f} MHz" if f else f", channel {ch:#x}"))
        for (ch, ga, sa), n in self.grants.most_common(12):
            f = self.freq(ch); sl = self.slot(ch); print(f"  voice grant x{n}: talkgroup {ga}, from unit {sa}, channel " + (f"{f:.4f} MHz" if f else f"{ch:#x}") + (f" slot {sl} (phase 2)" if sl is not None else ""))
        if self.opts:
            print("  talkgroups with service options (the grant's own flags: 'encrypted' is bit 0x40 of the service options byte, as set by the system, per talkgroup):")
            for ga, c in sorted(self.opts.items()):
                o, n = c.most_common(1)[0]
                print(f"    talkgroup {ga:5d}: options {o:#04x}  {'ENCRYPTED' if o & 0x40 else 'clear'}{'  emergency' if o & 0x80 else ''}  priority {o & 7}  ({sum(c.values())} grants" + (f", options vary: {dict(c)}" if len(c) > 1 else "") + ")")
        if self.other:
            print("  other:", dict(self.other.most_common(8)))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file"); ap.add_argument("--fs", type=float, required=True, help="[MS/s]"); ap.add_argument("--cf", type=float, required=True, help="[MHz]")
    ap.add_argument("--freq", type=float, nargs="+", required=True, help="channels [MHz]"); ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--corr", type=float, default=0.9)
    ap.add_argument("--tsbk", action="store_true", help="also decode the signalling blocks of TSDU frames and report CRC passes")
    a = ap.parse_args(argv)
    fs = a.fs*1e6; x = np.fromfile(a.file, np.complex64, count=int(fs*a.seconds))
    for f in a.freq:
        sn, c = symbols(x, fs, a.cf*1e6, f*1e6)
        hits, r = sync_hits(sn, a.corr)
        if not hits:
            print(f"{f:9.4f} MHz: no P25 sync (best correlation {r.max():.2f})"); continue
        nacs, duids = Counter(), Counter(); nid_ok = 0
        for i in hits:
            v = nid(refine(symbols.stream, i), 0)
            if v:
                nacs[v[0]] += 1; duids[v[1]] += 1; nid_ok += 1
        if a.tsbk:
            tot = ok = 0; ops = Counter(); T = Tsbk()
            for i in hits:
                fr = refine(symbols.stream, i)
                v = nid(fr, 0)
                if not v or v[1] != 0x7:
                    continue
                for by, good in tsbk_blocks(frame_bits(fr, 0)):
                    tot += 1; ok += good
                    if good:
                        ops[by[0] & 0x3f] += 1; T.add(by)
            print(f"{f:9.4f} MHz: TSBK blocks {tot}, CRC ok {ok}")
            T.report()
        gaps = np.diff(np.array(hits)/4800.0)
        print(f"{f:9.4f} MHz: {len(hits)} sync hits in {a.seconds:g} s (best corr {r.max():.2f}), median spacing {np.median(gaps)*1e3:.0f} ms; NID decoded {nid_ok}/{len(hits)}; NAC "
              f"{[(hex(k), v) for k, v in nacs.most_common(3)]}, DUID {[(DUID.get(k, hex(k)), v) for k, v in duids.most_common(4)]}")


if __name__ == "__main__":
    main()
