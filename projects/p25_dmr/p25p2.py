"""APCO P25 phase 2 (two-slot TDMA, H-DQPSK downlink, 6000 baud) channel decoder: symbols, slots, ISCH, DUID, descrambling, MAC messages, ESS and voice frames

    python projects/p25_dmr/p25p2.py data/survey/w852.7.cf64 --fs 4 --cf 852.7 --freq 853.8125 --wacn 0xbee00 --sysid 0x8d0 --nac 0x8d1 --seconds 20

Everything follows TIA-102.BBAB (physical layer) and TIA-102.BBAC (MAC layer, Annex E burst bit locations), read from the standards, not from another decoder.
  demod      the channel is mixed to zero, low-passed, resampled to 48 kS/s (8 per symbol) and the carrier offset found (the PSD centre, then the pi/4 fine step: differential phases to the 4th
             power point at -1); the phase of each symbol relative to the previous one, in units of pi/4, is the symbol (+-1, +-3), bits as in phase 1 (+3 = 01, +1 = 00, -1 = 10, -3 = 11)
  timing     the best of 8 sample phases is the one at which the 20-symbol inter-slot sync (VCH S-ISCH, Table 5-1) correlates best; its hits give the slot grid (180 symbols per slot, the
             ISCH is the last 10 symbols of one slot and the first 10 of the next)
  slot       DUID (8,4,4), ISCH (40 bits; I-ISCH is a (40,9,16) coset code: slot position in the superframe, logical channel, ultraframe count), then the bits of the burst
  scrambling the 44-bit LFSR from WACN / system ID / colour code, restarted at each superframe; slot k of the superframe uses sequence bits 360k .. 360k+359
  MAC        the I-OEMI / S-OEMI Reed-Solomon (63,35,29) codeword over GF(64) (shortened / punctured) is decoded with erasures, the MAC PDU opcode and CRC-12 checked: PTT (algorithm, key
             ID, message indicator, source, group), END_PTT, IDLE, ACTIVE, HANGTIME and their Group Voice Channel User messages
  ESS        the (44,16,29) code over the four 4V bursts (ESS-B) and the 2V burst (ESS-A): algorithm ID, key ID, message indicator
  voice      the 72-bit voice frames (c0..c3 of the half-rate vocoder; p25p2_voice.py decodes them)
Meant to be upstreamed into RPPS.
"""
import argparse
import os
import sys
from collections import Counter
from fractions import Fraction

import numpy as np
from scipy.signal import firwin, lfilter, resample_poly

import rpps as rp

import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); import _paths  # noqa: E402,F401
import burst_tones

BAUD = 6000
# ---- constants from the standard ---------------------------------------------------------------------------------------------------------------------------------------------------------
SYNC_VCH = np.array([3, 3, 3, -3, 3, 3, -3, 3, 3, 3, 3, -3, -3, -3, 3, -3, -3, -3, -3, -3], float) # S-ISCH sync on a voice channel, Table 5-1 (checked against the 0x575D57F7FF word)
C0_ISCH = 0x1842_29D4_61
ISCH_G = [  # Table 5-4, generator rows for info bits 8..0
    "1000100000010110110011100011011011010111", "0010000000011101111111010100111101100100", "0001000000001111010010110001011101011000", "0000110000000000110111101101000110001110",
    "0000001000001000000000111111011111111111", "0000100100000100100011011001101101110010", "0000000010011101101000111010000101110001", "0000000001011000110010111010101001001110",
    "0000000000110100001111011000010110010111"]
DUID_G = ["10001101", "01001011", "00101110", "00010111"] # Table 5-2
DUID_TYPE = {0b0000: "4V", 0b0110: "2V", 0b0011: "SACCH scrambled", 0b1001: "FACCH scrambled", 0b1100: "SACCH", 0b1111: "FACCH"}
# bit pairs (bit1, bit0) of the 36 symbols of a voice frame as (codeword number, bit index): Annex E, tables E-5 / E-7
_VF = ["0,23 0,5", "1,10 2,3", "0,22 0,4", "1,9 2,2", "0,21 0,3", "1,8 2,1", "0,20 0,2", "1,7 2,0", "0,19 0,1", "1,6 3,13", "0,18 0,0", "1,5 3,12", "0,17 1,22", "1,4 3,11", "0,16 1,21", "1,3 3,10",
       "0,15 1,20", "1,2 3,9", "0,14 1,19", "1,1 3,8", "0,13 1,18", "1,0 3,7", "0,12 1,17", "2,10 3,6", "0,11 1,16", "2,9 3,5", "0,10 1,15", "2,8 3,4", "0,9 1,14", "2,7 3,3", "0,8 1,13", "2,6 3,2",
       "0,7 1,12", "2,5 3,1", "0,6 1,11", "2,4 3,0"]
VF_MAP = [tuple(int(v) for v in p.split(",")) for s in _VF for p in s.split()]
VOICE_SYMS = {"4V": [11, 48, 96, 133], "2V": [11, 48]} # first symbol of each voice frame in the 180-symbol burst
GAP = 10 # symbols of ISCH half at each end of a slot


# ---- Reed-Solomon codes: (63,35,29) over GF(64), roots alpha^1 .. alpha^28 (Annex A), and the codes derived from it (config/coding/block/rs/p25p2_*)
_RS_ESS = rp.coding.load("block", "rs/p25p2_ess_44_16")._dec
_RS_EMI = {(26, 45, 9): rp.coding.load("block", "rs/p25p2_soemi")._dec, (30, 52, 6): rp.coding.load("block", "rs/p25p2_ioemi")._dec}


def emi_decode(bits, k_info, n_total, erase_tail):
    """Decode a shortened / punctured derived code: bits = k_info + parity hexbits as sent (info first, MSB first), n_total hexbits sent; the code is shortened to the front (35 - k_info zeros)
    and punctured at the end (erase_tail hexbits, erased). Returns (info bits as a list of 0/1 of length 6*k_info, corrected symbols) or (None, -1)"""
    code = _RS_EMI[(k_info, n_total, erase_tail)]
    info = code.decode(np.asarray(bits[:6*n_total], np.uint8))
    if code.errors[0] < 0:
        return None, -1
    return [int(b) for b in info], int(code.errors[0])


# ---- bit helpers, ISCH and DUID codes ------------------------------------------------------------------------------------------------------------------------------------------------------
def _bits(v, n):
    return [(v >> (n - 1 - i)) & 1 for i in range(n)]


_ISCH_CW = {}
for _i in range(128): # info bits 8..7 are reserved zero
    _v = np.zeros(40, np.uint8)
    for _r, _b in enumerate(_bits(_i, 9)):
        if _b:
            _v ^= np.array([int(c) for c in ISCH_G[_r]], np.uint8)
    _v ^= np.array(_bits(C0_ISCH, 40), np.uint8)
    _ISCH_CW[_i] = _v
_ISCH_MAT = np.array([_ISCH_CW[i] for i in range(128)])
_DUID_CW = {}
for _i in range(16):
    _v = np.zeros(8, np.uint8)
    for _r, _b in enumerate(_bits(_i, 4)):
        if _b:
            _v ^= np.array([int(c) for c in DUID_G[_r]], np.uint8)
    _DUID_CW[_i] = _v
_DUID_MAT = np.array([_DUID_CW[i] for i in range(16)])
_SYNC_BITS = np.array([b for s in SYNC_VCH for b in ((0, 1) if s > 2 else (0, 0) if s > 0 else (1, 0) if s > -2 else (1, 1))], np.uint8)


def decode_isch(bits40):
    """('S', distance) for the voice-channel sync, or ('I', dict(chan, loc, fr, count, slot), distance) for an information ISCH, or None. slot = position of the slot after this ISCH in the superframe"""
    ds = int(np.sum(bits40 != _SYNC_BITS))
    d = np.sum(_ISCH_MAT != bits40[None, :], axis=1); i = int(np.argmin(d))
    if ds <= 6 and ds < d[i]:
        return ("S", ds)
    if d[i] <= 7:
        chan, loc, fr, cnt = (i >> 5) & 3, (i >> 3) & 3, (i >> 2) & 1, i & 3
        return ("I", dict(chan=chan, loc=loc, fr=fr, count=cnt, slot=4*loc + chan), int(d[i]))
    return None


def decode_duid(bits8):
    d = np.sum(_DUID_MAT != bits8[None, :], axis=1); i = int(np.argmin(d))
    return (i, int(d[i])) if d[i] <= 1 else (None, int(d[i]))


# ---- scrambling ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def scramble_sequence(wacn, sysid, nac, n=4320):
    """The outbound scramble sequence (rpps scram/adt/p25p2): 44-bit LFSR from WACN / system ID / colour code, one superframe of n bits"""
    return rp.scram.load("p25p2", wacn=wacn, sysid=sysid, nac=nac).sequence(n)


# ---- demodulation -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def baseband48(x, fs, cf, freq):
    """Complex baseband of the channel at freq, 8 kHz low-pass, 48 kS/s"""
    z, fs1 = burst_tones.channel(x, fs, cf, freq)
    z = lfilter(firwin(121, 8e3, fs=fs1), 1, z)[120:]
    q = Fraction(48000, int(round(fs1)))
    return resample_poly(z, q.numerator, q.denominator).astype(np.complex64)


def carrier_offset(z):
    """Residual carrier offset [Hz]: the spectral centre of the (symmetric) signal, refined by the pi/4-DQPSK 4th-power line (the differential phases to the 4th power are all at -1)"""
    n = 2**13; k = len(z)//n
    S = (np.abs(np.fft.fftshift(np.fft.fft(z[:k*n].reshape(k, n)*np.hanning(n), axis=1), axes=1))**2).mean(0); f = np.fft.fftshift(np.fft.fftfreq(n, 1/48000))
    m = np.abs(f) < 9000; w = S[m]; w = w - 0.5*np.percentile(w, 90)
    w[w < 0] = 0
    coarse = float(np.sum(f[m]*w)/np.sum(w))
    z0 = z*np.exp(-2j*np.pi*coarse*np.arange(len(z))/48000)
    best = None
    for p in range(8):
        d = z0[p + 8::8]*np.conj(z0[p:-8:8])
        v = np.mean(d**4/np.abs(d)**4)
        if best is None or abs(v) > best[0]:
            best = (abs(v), np.angle(v))
    # 4*delta = angle + pi (mod 2 pi): delta per symbol, then Hz
    delta = (best[1] - np.pi)/4.0
    fine = (delta/(2*np.pi))*BAUD
    cand = [fine + k*BAUD/4 for k in range(-3, 4)] # ambiguity of a quarter of the symbol rate
    off = min(cand, key=lambda c: abs(c))
    return coarse + off, best[0]


def symbols(z, p):
    """Symbol levels (nominally +-1, +-3) at sample phase p: the phase step between successive symbols in units of pi/4"""
    d = z[p + 8::8]*np.conj(z[p:-8:8])
    return np.angle(d)/(np.pi/4)


def slice_sym(a):
    return np.where(a > 2, 3, np.where(a > 0, 1, np.where(a > -2, -1, -3)))


def sym_bits(levels):
    """levels (+-1, +-3) -> bits array (2 per symbol)"""
    out = np.zeros(2*len(levels), np.uint8)
    out[0::2] = levels < 0
    out[1::2] = np.abs(levels) > 2
    return out


def find_slots(a_by_phase):
    """Pick the sample phase and the slot grid: (phase, symbols levels as floats, positions of S-ISCH sync hits). a_by_phase: list of 8 level arrays"""
    best = None
    for p, a in enumerate(a_by_phase):
        if len(a) < 400:
            continue
        r = np.correlate(a, SYNC_VCH, "valid")/np.sqrt(np.maximum(1e-9, np.convolve(a**2, np.ones(20), "valid"))*np.sum(SYNC_VCH**2))
        score = np.sort(r)[-8:].mean()
        if best is None or score > best[0]:
            best = (score, p, r)
    return best


class Burst:
    """One decoded slot"""
    def __init__(self, start, bits, isch_prev):
        self.start = start; self.bits = bits; self.isch_prev = isch_prev
        self.slot = None; self.duid = None; self.kind = None; self.isch = None


def burst_fields(bits):
    """Raw (unscrambled) duid dibits positions in a 360-bit slot: symbols 10, 47, 132, 169"""
    return np.concatenate([bits[2*s:2*s + 2] for s in (10, 47, 132, 169)])


_CRC12 = rp.encap.load("crc", "crc12_p25")._enc

def crc12_ok(pdu_bits):
    """CRC-12 (x^12+x^11+x^7+x^4+x^2+x+1, inverted remainder) in the last 12 bits of the PDU"""
    return _CRC12.syndrome(pdu_bits) == 0


def mac_pdu(bits_info):
    """Parse a MAC PDU (info bits: header octet ... CRC-12): dict(ok, opcode, name, ...)"""
    n = len(bits_info)
    v = int("".join(str(int(b)) for b in bits_info), 2)
    def f(hi, lo):  # bit field by octet bit positions counted from the start: bits hi..lo of the PDU as byte offsets
        return (v >> (n - 1 - hi)) & ((1 << (hi - lo + 1)) - 1)
    ok = crc12_ok(bits_info)
    opcode = (v >> (n - 3)) & 7; offset = (v >> (n - 6)) & 7
    names = {1: "MAC_PTT", 2: "MAC_END_PTT", 3: "MAC_IDLE", 4: "MAC_ACTIVE", 6: "MAC_HANGTIME"}
    out = dict(ok=ok, opcode=opcode, name=names.get(opcode, f"opcode {opcode}"), offset=offset)
    def octets(a, b):  # value of octets a..b (inclusive) of the PDU
        return (v >> (n - 8*(b + 1))) & ((1 << (8*(b - a + 1))) - 1)
    if opcode == 1: # PTT: MI octets 1-9 (72 bits... octets 2-10?), see Figure 8-5: octets 1-9 hold the message indicator, 10 the algorithm ID, 11-12 the key ID, 13-15 source, 16-17 group
        out.update(mi=octets(1, 9), algid=octets(10, 10), kid=octets(11, 12), source=octets(13, 15), group=octets(16, 17))
    elif opcode == 2:
        out.update(cc=octets(2, 3) >> 4, source=octets(13, 15), group=octets(16, 17))
    elif opcode in (3, 4, 6):
        # messages from octet 1: B1 B2 MCO, with the lengths of Table 8-2 (the ones used on a voice channel)
        msgs = []; i = 1; end = (n - 12)//8
        while i < end:
            b1, b2, mco = (octets(i, i) >> 7) & 1, (octets(i, i) >> 6) & 1, octets(i, i) & 0x3F
            ln = {(0, 0, 0x00): None, (0, 0, 0x01): 7, (0, 0, 0x21): 14, (0, 0, 0x02): 8, (0, 0, 0x22): 15, (0, 0, 0x03): 7}.get((b1, b2, mco))
            if ln is None:
                break
            if (b1, b2, mco) == (0, 0, 0x01):
                msgs.append(dict(type="group voice user", svc=octets(i + 1, i + 1), group=octets(i + 2, i + 3), source=octets(i + 4, i + 6)))
            elif (b1, b2, mco) == (0, 0, 0x21):
                msgs.append(dict(type="group voice user", svc=octets(i + 1, i + 1), group=octets(i + 2, i + 3), source=octets(i + 4, i + 6)))
            else:
                msgs.append(dict(type=f"message {b1}{b2}/{mco:#x}"))
            i += ln
        out["msgs"] = msgs
    return out


# ---- slot decoding --------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def _sym_range(*ranges):
    return [s for a, b in ranges for s in range(a, b)]


_SYM_SOEMI = _sym_range((11, 47), (48, 79), (100, 132), (133, 169))
_SYM_IOEMI = _sym_range((11, 47), (48, 132), (133, 169))
_SYM_ESSA = _sym_range((84, 132), (133, 169))
LCH_OF_SLOT = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0] # Figure 5-3: logical channel of each outbound slot of the superframe
S_ISCH_SLOTS = {2, 3, 6, 7, 10, 11}


def _syms_bits(bits, syms):
    return np.concatenate([bits[2*s:2*s + 2] for s in syms])


def voice_codewords(bits, kind):
    """The 72-bit voice frames of a 4V / 2V burst as lists [c0 (24 bits), c1 (23), c2 (11), c3 (14)], each a list of bits MSB (index n-1) first"""
    out = []
    for s0 in VOICE_SYMS[kind]:
        fb = bits[2*s0:2*s0 + 72]
        c = [[0]*24, [0]*23, [0]*11, [0]*14]
        for k, (ci, bi) in enumerate(VF_MAP):
            c[ci][len(c[ci]) - 1 - bi] = int(fb[k])
        out.append(c)
    return out


class Slot:
    def __init__(self, t, pos, raw):
        self.t = t; self.pos = pos; self.raw = raw; self.duid = None; self.kind = None; self.isch = None
        self.pdu = None; self.voice = None; self.ess_b = None; self.ess_a = None; self.lch = None; self.scrambled = None


class Decoder:
    """Feeds on symbol-level streams; keeps the superframe position (from I-ISCH), the ESS assembly per logical channel and statistics"""
    def __init__(self, wacn, sysid, nac):
        self.seq = scramble_sequence(wacn, sysid, nac)
        self.delta = Counter(); self.stats = Counter(); self.ess_parts = {0: [], 1: []}; self.slots = []
        self.last = None  # (absolute start sample of the last slot, its running slot number)

    def slot_number(self, abs_start):
        """Running slot number of a slot from its absolute start (48 kS/s samples): continuity with the previous slot, 1440 samples apart"""
        if self.last is None:
            n = 0
        else:
            n = self.last[1] + int(round((abs_start - self.last[0])/1440.0))
        self.last = (abs_start, n)
        return n

    def slot_position(self, n):
        d = self.delta.most_common(1)
        return None if not d else (n + d[0][0]) % 12

    def decode_slots(self, a, hit0, abs0=0, after=-1):
        """Decode every slot of the level array a on the grid defined by an S-ISCH sync hit at symbol hit0 (the slot after it starts at hit0 + 10). Returns the list of Slot"""
        first = (hit0 + 10) % 180
        starts = list(range(first if first >= GAP else first + 180, len(a) - 180 + 1, 180))
        out = []; raws = []
        for s in starts:
            ab = abs0 + 8*s
            if ab <= after:
                continue  # already decoded from an earlier window
            lv = slice_sym(a[s - GAP:s + 180])
            bits = sym_bits(lv)  # 20 + 360 bits
            raws.append((s, bits, self.slot_number(ab)))
        # slot position from I-ISCH: slot after an ISCH is 4*loc + chan
        for i, (s, bits, n) in enumerate(raws):
            isch = decode_isch(np.concatenate([bits[:20], bits[20:40]]))  # last 10 symbols of the previous slot, first 10 of this one
            if isch and isch[0] == "I":
                self.delta[(isch[1]["slot"] - n) % 12] += 1
        for i, (s, bits, n) in enumerate(raws):
            b = bits[20:]  # the slot's 360 bits
            sl = Slot(abs0 + 8*s, self.slot_position(n), b)
            sl.isch = decode_isch(np.concatenate([bits[:20], b[:20]]))
            # NB the ISCH is bits[0:20] (prev slot, symbols 170-179) + b[0:20] (this slot, symbols 0-9)
            duid, dist = decode_duid(burst_fields(b))
            sl.duid = duid
            sl.kind = DUID_TYPE.get(duid, "?")
            self.stats[sl.kind] += 1
            if duid is None or sl.pos is None:
                out.append(sl); continue
            scr = sl.kind in ("4V", "2V", "SACCH scrambled", "FACCH scrambled")
            sl.scrambled = scr
            bb = b ^ self.seq[360*sl.pos:360*sl.pos + 360] if scr else b
            sl.lch = LCH_OF_SLOT[sl.pos]
            if sl.kind in ("FACCH", "FACCH scrambled", "SACCH", "SACCH scrambled"):
                if sl.kind.startswith("FACCH"):
                    bitsx = _syms_bits(bb, _SYM_SOEMI); info, ne = emi_decode(bitsx, 26, 45, 9); n_info = 156
                else:
                    bitsx = _syms_bits(bb, _SYM_IOEMI); info, ne = emi_decode(bitsx, 30, 52, 6); n_info = 180
                self.stats[sl.kind + (" rs ok" if info is not None else " rs fail")] += 1
                if info is not None:
                    sl.pdu = mac_pdu(info); sl.pdu["rs_errors"] = ne
                    self.stats[sl.kind + (" crc ok" if sl.pdu["ok"] else " crc bad")] += 1
            elif sl.kind in ("4V", "2V"):
                sl.voice = voice_codewords(bb, sl.kind)
                if sl.kind == "4V":
                    sl.ess_b = [int(v) for v in bb[2*84:2*96]]
                else:
                    sl.ess_a = [int(v) for v in _syms_bits(bb, _SYM_ESSA)]
                self.ess_collect(sl)
            out.append(sl)
        self.slots += out
        return out

    def ess_collect(self, sl):
        parts = self.ess_parts[sl.lch]
        if sl.kind == "4V":
            parts.append(sl.ess_b)
            del parts[:-4]
        else:
            if len(parts) == 4:
                hx = [int("".join(map(str, p[6*i:6*i + 6])), 2) for p in parts for i in range(4)] + [int("".join(map(str, sl.ess_a[6*i:6*i + 6])), 2) for i in range(28)]
                info = _RS_ESS.decode(np.array(hx), symbols=True)
                ok = _RS_ESS.errors[0] >= 0
                self.stats["ess rs ok" if ok else "ess rs fail"] += 1
                if ok:
                    bits = "".join(format(int(h), "06b") for h in info)
                    sl.ess = dict(algid=int(bits[:8], 2), kid=int(bits[8:24], 2), mi=int(bits[24:], 2))
            self.ess_parts[sl.lch] = []


class ChanP2:
    """A phase 2 channel inside the monitor: complex 48 kS/s baseband in (from the channelizer), decoded slots out (via on_slots) every 0.5 s on a rolling 1.5 s window.
    params() gives (WACN, system ID, colour code) for descrambling, or None while they are not known yet"""
    def __init__(self, freq, kc, params, on_slots):
        self.freq = freq; self.kc = kc; self.params = params; self.on_slots = on_slots; self.role = "voice"; self.kind = "p2"
        self.taps = firwin(121, 8e3, fs=48000); self.zi = np.zeros(120, np.complex64)
        self.z = np.zeros(0, np.complex64); self.base = 0; self.since = 0; self.dec = None; self.key = None; self.last_abs = -1
        self.level = -120.0; self.active = False; self.frames = 0; self.nac = None; self.last_frame = 0.0; self.score = 0.0; self.slots_ok = 0

    def push(self, y):
        z, self.zi = lfilter(self.taps, 1, y, zi=self.zi)
        self.z = np.concatenate((self.z, z.astype(np.complex64)))
        if len(self.z) > 3*48000:
            cut = len(self.z) - 3*48000; self.z = self.z[cut:]; self.base += cut
        self.since += len(y)
        if self.since >= 24000:
            self.since = 0; self.tick()

    def tick(self):
        import time
        n = min(len(self.z), 72000)
        if n < 24000:
            return
        seg = self.z[-n:]; base = self.base + len(self.z) - n
        p = np.abs(seg)**2; self.level = float(10*np.log10(np.percentile(p, 90) + 1e-20)); self.active = bool(np.percentile(p, 90) > 8*np.percentile(p, 10))
        par = self.params()
        if par is None:
            return
        if self.key != par:
            self.dec = Decoder(*par); self.key = par; self.last_abs = -1
        off, mag = carrier_offset(seg)
        if mag < 0.3:
            return  # no pi/4-DQPSK carrier in the window
        seg = seg*np.exp(-2j*np.pi*off*np.arange(len(seg))/48000)
        A = [symbols(seg, ph) for ph in range(8)]
        best = find_slots(A)
        self.score = best[0] if best else 0.0
        if best is None or best[0] < 0.8:
            return
        sc, ph, r = best; a = A[ph]
        hits = [i for i in range(1, len(r) - 1) if r[i] > 0.8 and r[i] >= r[i - 1] and r[i] >= r[i + 1]]
        if not hits:
            return
        sl = self.dec.decode_slots(a, hits[0], abs0=base + ph + 8, after=self.last_abs)
        if sl:
            self.last_abs = sl[-1].t
            good = [s for s in sl if s.kind != "?"]
            self.frames += len(good); self.slots_ok += len(good)
            if good:
                self.last_frame = time.monotonic()
            self.on_slots(self, sl)
