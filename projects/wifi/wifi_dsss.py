"""802.11b (DSSS, 1 and 2 Mbit/s) receiver: bursts in a 20 MS/s capture -> PLCP -> MAC frames

    python projects/wifi/wifi_dsss.py data/wifi_ch11.cf64 --fs 20 --channel 11

IEEE 802.11-2020 clause 16. At 11 Mchip/s each bit (1 Mbit/s DBPSK) or dibit (2 Mbit/s DQPSK) is spread by the 11-chip Barker code +1 -1 +1 +1 -1 +1 +1 +1 -1 -1 -1.
Long preamble: 128 scrambled ones (SYNC), SFD 0xF3A0, then the PLCP header at 1 Mbit/s: SIGNAL (rate/0.1 Mbit/s: 0x0A = 1, 0x14 = 2),
SERVICE, LENGTH (microseconds of the PSDU), CRC-16; then the PSDU (a MAC frame ending in a CRC-32), all bytes LSB first. Every bit is scrambled by the
self-synchronising polynomial z^-7 + z^-4 + 1. Differential modulation means the unknown carrier phase does not matter, only a rotation per symbol
(the carrier offset: a HackRF is ~15 ppm = 36 kHz off at 2.4 GHz, 0.23 rad per symbol), which is estimated from the preamble.
Steps per burst (each has a statistic that is chance when wrong): timing from despread energy over the SYNC; carrier offset from the squared
differential symbols; the SFD (16 bits, chance 1.5e-5); the header CRC-16 (chance 1.5e-5); the frame CRC-32 (chance 2e-10).
"""
import argparse
import zlib

import numpy as np
from scipy.signal import resample_poly

import rpps as rp

BARKER = np.array([1, -1, 1, 1, -1, 1, 1, 1, -1, -1, -1], float)
OS = 4 # samples per chip after resampling to 44 MS/s
TEMPLATE = np.repeat(BARKER, OS)
SPS = 11*OS # samples per symbol

def to_44(x, fs):
    from fractions import Fraction
    r = Fraction(44e6/fs).limit_denominator(1000)
    return resample_poly(x, r.numerator, r.denominator)

def bursts(x, fs, factor=6.0, win=100, min_us=150):
    """(start, end) sample ranges where the power stays above `factor` x the 20th percentile (the noise floor)"""
    p = np.abs(x)**2
    e = p[:len(p)//win*win].reshape(-1, win).mean(1)
    on = e > factor*np.percentile(e, 20)
    d = np.diff(np.r_[0, on.astype(int), 0])
    st = np.where(d == 1)[0]*win; en = np.where(d == -1)[0]*win
    return [(s, e_) for s, e_ in zip(st, en) if (e_ - s)/fs*1e6 >= min_us]

def despread(y, start, n, off=0.0):
    """Correlate n consecutive symbols from start (float sample position): returns complex symbol values"""
    out = np.empty(n, complex)
    base = int(np.floor(start)); frac = start - base
    for k in range(n):
        i = base + k*SPS
        seg = y[i:i + SPS + 1]
        if len(seg) < SPS + 1:
            return out[:k]
        if frac:
            seg = seg[:-1]*(1 - frac) + seg[1:]*frac
        else:
            seg = seg[:SPS]
        out[k] = seg @ TEMPLATE
    return out

def find_timing(y, s, e, nsym=100):
    """Symbol-boundary phase (0..SPS-1) maximizing the despread energy over `nsym` symbols from about 5 symbols into the burst"""
    best = (0, 0.0)
    for ph in range(SPS):
        c = despread(y, s + 5*SPS + ph, nsym)
        if len(c) < nsym:
            continue
        en = float(np.sum(np.abs(c)**2))
        if en > best[1]:
            best = (ph, en)
    return best

def track(y, start, nsym):
    """Symbol values along a burst, following the timing: every 40 symbols re-centre by comparing early, prompt and late despreading (+-1 sample)"""
    out = []; pos = float(start)
    k = 0
    while k < nsym:
        n = min(40, nsym - k)
        c = [despread(y, pos + d, n) for d in (-1.0, 0.0, 1.0)]
        if len(c[1]) < n:
            break
        en = [float(np.sum(np.abs(v)**2)) for v in c]
        out.append(c[1])
        if en[0] > en[1]*1.02 and en[0] >= en[2]:
            pos -= 0.5
        elif en[2] > en[1]*1.02:
            pos += 0.5
        pos += n*SPS
        k += n
    return np.concatenate(out) if out else np.zeros(0, complex)

def descramble(bits):
    b = np.asarray(bits, np.uint8)
    out = np.zeros(len(b), np.uint8)
    for n in range(7, len(b)):
        out[n] = b[n] ^ b[n - 4] ^ b[n - 7]
    return out

SFD_BITS = [(0xF3A0 >> i) & 1 for i in range(16)] # transmitted LSB first

_PLCP_CRC = rp.encap.load("crc", "crc16_genibus")._enc # CRC-16 CCITT, ones start, ones' complement (15.3.3.7)

def crc16_plcp(bits):
    """CRC-16 of the 32 PLCP header bits (SIGNAL, SERVICE, LENGTH), as 16 bits MSB first"""
    return [int(b) for b in _PLCP_CRC.to_bits(_PLCP_CRC.compute_bits(bits))]

def bytes_lsb(bits):
    n = len(bits)//8
    b = np.asarray(bits[:n*8], np.uint8).reshape(-1, 8)
    return bytes((b*(1 << np.arange(8))).sum(1).astype(np.uint8))

def decode_burst(y, s, e, fs=44e6):
    """One burst -> dict(rate, length_us, frame bytes, fcs_ok, snr_proxy) or None"""
    ph, en = find_timing(y, s, e)
    nsym_total = int((e - s)/SPS)
    if nsym_total < 192 + 48:
        return None
    start = s + 5*SPS + ph
    z = track(y, start, nsym_total - 6)
    if len(z) < 240:
        return None
    d = z[1:]*np.conj(z[:-1]) # differential: bit 1 = phase flip
    # carrier offset: BPSK differential symbols squared have a constant angle = twice the per-symbol rotation (SYNC portion)
    sync = d[:100]
    rot = np.angle(np.sum(sync**2))/2
    for pol in (0, 1): # the squaring leaves a pi ambiguity: the rotation or rotation + pi
        r = rot + pol*np.pi
        dd = d*np.exp(-1j*r)
        bits = (dd.real < 0).astype(np.uint8)
        sc = descramble(bits)
        # find the SFD after a run of ones
        s_arr = sc
        for i in range(20, min(len(s_arr) - 64, 260)):
            if np.array_equal(s_arr[i:i + 16], SFD_BITS) and s_arr[max(0, i - 16):i].sum() >= 14:
                h = s_arr[i + 16:i + 16 + 48]
                if len(h) < 48:
                    continue
                if crc16_plcp(h[:32]) != list(h[32:48]):
                    continue
                signal = int(bytes_lsb(h[:8])[0]); length_us = int.from_bytes(bytes_lsb(h[16:32]), "little")
                rate = {0x0A: 1, 0x14: 2}.get(signal)
                res = dict(signal=signal, rate=rate, length_us=length_us, start=s, polarity=pol, rot=rot, sym=i + 16 + 48)
                if rate == 1:
                    nbytes = length_us//8
                    pb = s_arr[i + 64:i + 64 + nbytes*8]
                elif rate == 2:
                    nbytes = length_us//4
                    k0 = i + 64
                    # DQPSK: the dibit comes from the differential phase; the carrier rotation is already removed
                    ph_ = np.angle(dd[k0:k0 + nbytes*4])
                    q = np.round(ph_/(np.pi/2)).astype(int) % 4
                    dib = {0: (0, 0), 1: (0, 1), 2: (1, 1), 3: (1, 0)}
                    # (the scrambler runs over the dibits as a bit stream)
                    raw = np.array([dib[v] for v in q], np.uint8).reshape(-1)
                    scr_all = np.concatenate((bits[:k0], raw))
                    # descramble the 2 Mbit/s part with the register carried over from the 1 Mbit/s part
                    full = np.concatenate((bits[:k0], raw))
                    pb = descramble(full)[k0:k0 + nbytes*8]
                else:
                    return res
                data = bytes_lsb(pb)
                if len(data) >= 10:
                    res["frame"] = data
                    res["fcs_ok"] = len(data) > 4 and zlib.crc32(data[:-4]) == int.from_bytes(data[-4:], "little")
                return res
    return None

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("path"); ap.add_argument("--fs", type=float, default=20); ap.add_argument("--channel", type=int, default=0)
    a = ap.parse_args(argv)
    x = np.fromfile(a.path, np.complex64).astype(np.complex128)
    fs = a.fs*1e6
    bs = bursts(x, fs)
    print(f"{len(bs)} bursts of >= 150 us")
    y = to_44(x, fs)
    sc = 44e6/fs
    out = []
    for s, e in bs:
        r = decode_burst(y, int(s*sc), int(e*sc))
        if r:
            out.append(r)
    ok = [r for r in out if r.get("fcs_ok")]
    print(f"PLCP headers with a valid CRC-16: {len(out)}; frames with a valid FCS: {len(ok)}")
    return out

if __name__ == "__main__":
    main()
