"""Bluetooth LE advertising-channel receiver (1 Mbit/s GFSK): bursts in a capture -> link-layer packets -> advertising PDUs

    python projects/ble/ble_adv.py data/ble_37.cf64 --channel 37          # complex64 at 4 MS/s centred on the channel

Bluetooth Core Specification v6.0, Vol 6 Part B (link layer): 1 Msym/s GFSK, modulation index 0.5 (+-250 kHz), a bit 1 is the higher frequency. Packet on air:
preamble (0x55 here, 8 bits), access address 0x8E89BED6 (32 bits), PDU (2-byte header + payload), CRC-24; all fields LSB first. PDU and CRC are whitened with
x^7 + x^4 + 1, seeded with position 0 = 1 and positions 1..6 = the channel index (37, 38, 39 for 2402, 2426, 2480 MHz) MSB first. The CRC has polynomial
x^24 + x^10 + x^9 + x^6 + x^4 + x^3 + x + 1, preset 0x555555 (position 0 = LSB), computed over the PDU bits in transmitted order, sent MSB first (position 23 first).
Steps per burst, each with a statistic that is chance when wrong: FM discriminator, carrier offset removed (a HackRF is ~15 ppm = 36 kHz off, 14% of the deviation);
bit slicing at the best of four sample phases; the access address (32 bits, at most 2 wrong, chance ~1e-7 per position) preceded by an alternating preamble;
the CRC-24 (chance 6e-8). Only packets with a valid CRC count.
Advertising PDU header: type (4 bits: 0 ADV_IND, 1 ADV_DIRECT_IND, 2 ADV_NONCONN_IND, 3 SCAN_REQ, 4 SCAN_RSP, 5 CONNECT_IND, 6 ADV_SCAN_IND, 7 ADV_EXT_IND),
ChSel (bit 5), TxAdd (bit 6: 0 public / 1 random address), RxAdd (bit 7), length (8 bits).
"""
import argparse

import numpy as np

import rpps as rp

AA = 0x8E89BED6
AA_BITS = np.array([(AA >> i) & 1 for i in range(32)], np.uint8)
PDU_TYPES = {0: "ADV_IND", 1: "ADV_DIRECT_IND", 2: "ADV_NONCONN_IND", 3: "SCAN_REQ", 4: "SCAN_RSP", 5: "CONNECT_IND", 6: "ADV_SCAN_IND", 7: "ADV_EXT_IND"}
CHANNEL_INDEX = {37: 37, 38: 38, 39: 39}
FS = 4e6

def whiten_bits(bits, channel):
    """De/whiten a bit array (the same operation): rpps sequence/recurrence/ble_whiten (LFSR x^7 + x^4 + 1 seeded with the channel index)"""
    return rp.code.scram.load("ble_whiten", channel=int(channel)).encode(bits).astype(np.uint8)

_CRC24 = rp.link.load("crc", "crc24_ble")._enc

def crc24(bits, init=0x555555):
    """CRC-24 over `bits` in transmitted order; returns the 24 bits in the order they are sent"""
    if init != _CRC24.init:
        raise ValueError("only the advertising CRC init (0x555555) is set up")
    return [int(b) for b in _CRC24.to_bits(_CRC24.compute_bits(bits))]

def bursts(x, fs=FS, factor=6.0, win=8, min_us=60, max_us=3000):
    p = np.abs(x)**2
    e = p[:len(p)//win*win].reshape(-1, win).mean(1)
    on = e > factor*np.percentile(e, 20)
    d = np.diff(np.r_[0, on.astype(int), 0])
    st = np.where(d == 1)[0]*win; en = np.where(d == -1)[0]*win
    return [(s, e_) for s, e_ in zip(st, en) if min_us <= (e_ - s)/fs*1e6 <= max_us]

def demod(seg, fs=FS):
    """Instantaneous frequency (Hz) with the burst's mean removed, smoothed over one symbol"""
    f = np.angle(seg[1:]*np.conj(seg[:-1]))*fs/(2*np.pi)
    core = f[len(f)//10:len(f)*9//10]
    f = f - np.median(core)
    sps = int(round(fs/1e6))
    return np.convolve(f, np.ones(sps)/sps, mode="same")

def find_packet(f, sps=4, max_err=2):
    """Slice at each of the `sps` sample phases and look for the access address: returns (bits, index of the AA's first bit, errors) of the best, or None"""
    best = None
    for ph in range(sps):
        bits = (f[ph::sps] > 0).astype(np.uint8)
        if len(bits) < 32 + 16 + 24:
            continue
        # sliding Hamming distance to the access address
        n = len(bits) - 32
        win = np.lib.stride_tricks.sliding_window_view(bits, 32)[:n]
        err = np.sum(win != AA_BITS, axis=1)
        k = int(np.argmin(err))
        if err[k] <= max_err and k >= 8:
            pre = bits[k - 8:k]
            if np.all(pre[1:] != pre[:-1]) or err[k] == 0: # an alternating preamble
                if best is None or err[k] < best[2]:
                    best = (bits, k, int(err[k]), ph)
    return best

def decode_burst(seg, channel, fs=FS):
    """-> dict(pdu, header, payload, crc_ok, ...) for one burst, or None if no access address"""
    f = demod(seg, fs)
    hit = find_packet(f, int(round(fs/1e6)))
    if not hit:
        return None
    bits, k, err, ph = hit
    body = bits[k + 32:]
    if len(body) < 16 + 24:
        return None
    hdr = whiten_bits(body[:16], channel)
    # the whitener runs on through the PDU: de-whiten everything from the header on in one pass
    length = int(sum(int(hdr[8 + i]) << i for i in range(8)))
    total = 16 + 8*length + 24
    if length < 6 or len(body) < total: # every advertising PDU carries at least a 6-byte address
        return dict(crc_ok=False, aa_err=err, length=length)
    dw = whiten_bits(body[:total], channel)
    pdu, crc_rx = dw[:16 + 8*length], dw[16 + 8*length:]
    crc_ok = crc24(pdu) == [int(b) for b in crc_rx]
    payload = bytes(int(sum(int(pdu[16 + 8*j + i]) << i for i in range(8))) for j in range(length))
    h0 = int(sum(int(pdu[i]) << i for i in range(8)))
    return dict(crc_ok=crc_ok, aa_err=err, length=length, type=h0 & 15, chsel=(h0 >> 5) & 1, txadd=(h0 >> 6) & 1, rxadd=(h0 >> 7) & 1, payload=payload, phase=ph)

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("path"); ap.add_argument("--channel", type=int, required=True, choices=(37, 38, 39)); ap.add_argument("--fs", type=float, default=4.0)
    a = ap.parse_args(argv)
    x = np.fromfile(a.path, np.complex64).astype(np.complex128)
    fs = a.fs*1e6
    bs = bursts(x, fs)
    res = []
    for s, e in bs:
        r = decode_burst(x[max(0, s - 16):e + 16], a.channel, fs)
        if r:
            r["t"] = s/fs; res.append(r)
    ok = [r for r in res if r["crc_ok"]]
    print(f"{len(bs)} bursts; access address found in {len(res)}; CRC-24 valid in {len(ok)}")
    return res

if __name__ == "__main__":
    main()
