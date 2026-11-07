"""DVB-S2 physical layer (EN 302 307) helpers: start of frame, PLS header, frame sizes

Frames start with a 90 symbol header in pi/2-BPSK: a 26 symbol start of frame (SOF) and a 64 symbol PLS code that
says what follows (modulation, code rate, short/normal frame, pilots). The LDPC/BCH layer behind it
needs the standard's parity tables and is not here.
"""
import numpy as np

SOF = 0x18D2E82
PLS_SCRAMBLE = 0x719D83C953422DFA
# Generator rows of the (32,6) code, MSB-first columns, for bits (b6..b1) of the 6 coded PLS bits
_GEN = [0x55555555, 0x33333333, 0x0F0F0F0F, 0x00FF00FF, 0x0000FFFF, 0xFFFFFFFF]

MODCOD = {
    1: ("QPSK", "1/4"), 2: ("QPSK", "1/3"), 3: ("QPSK", "2/5"), 4: ("QPSK", "1/2"), 5: ("QPSK", "3/5"),
    6: ("QPSK", "2/3"), 7: ("QPSK", "3/4"), 8: ("QPSK", "4/5"), 9: ("QPSK", "5/6"), 10: ("QPSK", "8/9"),
    11: ("QPSK", "9/10"), 12: ("8PSK", "3/5"), 13: ("8PSK", "2/3"), 14: ("8PSK", "3/4"), 15: ("8PSK", "5/6"),
    16: ("8PSK", "8/9"), 17: ("8PSK", "9/10"), 18: ("16APSK", "2/3"), 19: ("16APSK", "3/4"), 20: ("16APSK", "4/5"),
    21: ("16APSK", "5/6"), 22: ("16APSK", "8/9"), 23: ("16APSK", "9/10"), 24: ("32APSK", "3/4"), 25: ("32APSK", "4/5"),
    26: ("32APSK", "5/6"), 27: ("32APSK", "8/9"), 28: ("32APSK", "9/10"),
}
BITS_PER_SYMBOL = {"QPSK": 2, "8PSK": 3, "16APSK": 4, "32APSK": 5}

def _bits(v, n):
    return np.array([(v >> (n-1-i)) & 1 for i in range(n)], dtype=np.uint8)

def bpsk_symbols(bits):
    """pi/2 BPSK: even symbols (1+j)(1-2b)/sqrt2, odd symbols (-1+j)(1-2b)/sqrt2"""
    d = 1.0 - 2.0*np.asarray(bits, dtype=float)
    k = np.arange(len(d))
    return np.where(k % 2 == 0, d + 1j*d, -d + 1j*d)/np.sqrt(2)

def bpsk_bits(sym):
    """Hard bits from pi/2 BPSK symbols"""
    k = np.arange(len(sym))
    d = np.where(k % 2 == 0, sym.real + sym.imag, -sym.real + sym.imag)
    return (d < 0).astype(np.uint8)

def sof_symbols():
    return bpsk_symbols(_bits(SOF, 26))

def pls_codeword(modcod, short=False, pilots=False):
    """The 64 transmitted PLS bits (before pi/2 BPSK) for a PLS value"""
    b = [(modcod >> (4-i)) & 1 for i in range(5)] + [int(short)]   # b1..b6 in transmission order
    y = np.zeros(32, dtype=np.uint8)
    for bit, row in zip(b, _GEN[::-1][:6] if False else _GEN):
        if bit:
            y ^= _bits(row, 32)
    c = np.zeros(64, dtype=np.uint8)
    c[0::2] = y
    c[1::2] = y ^ int(pilots)
    return c ^ _bits(PLS_SCRAMBLE, 64)

def decode_pls(bits):
    """Best PLS value for 64 received header bits: (modcod, short, pilots, bit errors)"""
    best = None
    for m in range(1, 32):
        for s in (0, 1):
            for p in (0, 1):
                e = int(np.sum(pls_codeword(m, bool(s), bool(p)) != bits))
                if best is None or e < best[3]:
                    best = (m, bool(s), bool(p), e)
    return best

def frame_symbols(modcod, short=False, pilots=False):
    """Symbols in one PLFRAME including the header and pilots"""
    mod, _ = MODCOD[modcod]
    n = (16200 if short else 64800)//BITS_PER_SYMBOL[mod]
    slots = n//90
    out = 90 + n
    if pilots:
        out += (slots - 1)//16 * 36
    return out

def find_frames(sym, thr=6.0):
    """Locate PLHEADERs in locked symbols (any 90 degree-ish carrier phase). Returns [(index, phase, correlation)]"""
    u = sof_symbols()
    y = sym/np.sqrt(np.mean(np.abs(sym)**2))
    c = np.correlate(y, u, "valid")
    e = np.sqrt(np.convolve(np.abs(y)**2, np.ones(len(u)), "valid")*np.sum(np.abs(u)**2))
    m = np.abs(c)/e
    peaks = np.nonzero(m > m.mean() + thr*m.std())[0]
    out = []
    for i in peaks:
        if out and i - out[-1][0] <= len(u):
            if m[i] > out[-1][2]:
                out[-1] = (int(i), float(np.angle(c[i])), float(m[i]))
            continue
        out.append((int(i), float(np.angle(c[i])), float(m[i])))
    return out

def read_header(sym, index, phase):
    """Decode the PLS of the frame whose SOF starts at `index` (a carrier phase from find_frames)"""
    h = sym[index + 26:index + 90]*np.exp(-1j*phase)
    h = h/np.sqrt(np.mean(np.abs(h)**2))
    return decode_pls(bpsk_bits(h))

def gold_code(n=0, length=2**18 - 1):
    """Scrambling sequence R_n(i) in {0,1,2,3} (EN 302 307 5.5.4): the symbol is multiplied by exp(j*pi/2*R)"""
    N = 2**18 - 1
    x = np.zeros(N, dtype=np.uint8)
    y = np.zeros(N, dtype=np.uint8)
    x[0] = 1
    y[:18] = 1
    for i in range(N - 18):
        x[i + 18] = x[i + 7] ^ x[i]
        y[i + 18] = y[i + 10] ^ y[i + 7] ^ y[i + 5] ^ y[i]
    i = np.arange(length)
    z = lambda k: x[(k + n) % N] ^ y[k % N]
    return 2*z((i + 131072) % N) + z(i)

def pl_scramble(payload, n=0, descramble=True):
    """(De)scramble the PLFRAME payload symbols (header excluded)"""
    R = gold_code(n, len(payload))
    return payload*np.exp((-1j if descramble else 1j)*np.pi/2*R)

def pilot_blocks(modcod, short=False):
    """(start, length) of the pilot blocks inside the payload (after the 90 symbol header)"""
    mod, _ = MODCOD[modcod]
    slots = (16200 if short else 64800)//BITS_PER_SYMBOL[mod]//90
    return [(16*90*(b + 1) + 36*b, 36) for b in range((slots - 1)//16)]

def parse_bbheader(bits):
    """BBHEADER (80 bits, after BB descrambling): fields and whether its CRC-8 checks out"""
    from .. import encap
    b = np.asarray(bits, dtype=np.uint8)[:80]
    by = np.packbits(b)
    crc = encap.load("crc", "crc8_dvbs2")._enc
    ts_gs = (by[0] >> 6) & 3
    return {
        "ts_gs": {3: "transport stream", 0: "generic packetized", 1: "generic continuous", 2: "reserved"}[int(ts_gs)],
        "sis": bool((by[0] >> 5) & 1), "ccm": bool((by[0] >> 4) & 1), "issyi": bool((by[0] >> 3) & 1),
        "npd": bool((by[0] >> 2) & 1), "rolloff": [0.35, 0.25, 0.20, None][int(by[0] & 3)],
        "upl": int(by[2] << 8 | by[3]), "dfl": int(by[4] << 8 | by[5]), "sync": int(by[6]),
        "syncd": int(by[7] << 8 | by[8]), "crc_ok": crc.compute(by[:9]) == int(by[9]),
    }

# 8PSK labels (b0 b1 b2 as an integer, b0 first) -> angle [rad]
PSK8 = np.exp(1j*np.array([np.pi/4, 0, np.pi, -3*np.pi/4, np.pi/2, 3*np.pi/4, -np.pi/4, -np.pi/2]))

def derotate_pilots(sym, modcod, short=False):
    """After PL descrambling: remove the carrier phase using the pilot blocks (known (1+j)/sqrt2). Returns (data, noise var)"""
    blocks = pilot_blocks(modcod, short)
    ref = (1 + 1j)/np.sqrt(2)
    centres, ph = [], []
    for s, l in blocks:
        centres.append(s + l/2)
        ph.append(np.angle(np.mean(sym[s:s+l]*np.conj(ref))))
    ph = np.unwrap(ph)
    phase = np.interp(np.arange(len(sym)), centres, ph)
    out = sym*np.exp(-1j*phase)
    err = np.concatenate([out[s:s+l] - ref for s, l in blocks])
    keep = np.ones(len(sym), dtype=bool)
    for s, l in blocks:
        keep[s:s+l] = False
    return out[keep], float(np.mean(np.abs(err)**2))

def demap_llr(sym, points, labels, var):
    """Max-log LLR (positive = bit 0) for symbols against `points` whose labels are the integers `labels`"""
    bps = int(np.log2(len(points)))
    d2 = np.abs(sym[:, None] - points[None, :])**2
    out = np.zeros((len(sym), bps))
    for b in range(bps):
        bit = (np.asarray(labels) >> (bps - 1 - b)) & 1
        out[:, b] = (d2[:, bit == 1].min(axis=1) - d2[:, bit == 0].min(axis=1))/var
    return out

# --- Whole frames ------------------------------------------------------------------------------------------

_RATE_FILE = {"1/4": "1_4", "1/3": "1_3", "2/5": "2_5", "1/2": "1_2", "3/5": "3_5", "2/3": "2_3", "3/4": "3_4",
              "4/5": "4_5", "5/6": "5_6", "8/9": "8_9", "9/10": "9_10"}
# Order in which the 8PSK bit interleaver's columns are read: reversed for rate 3/5
_COLS8 = {"3/5": (2, 1, 0)}

def _qpsk_llr(sym, var):
    s = np.sqrt(2)
    return np.stack((s*sym.real, s*sym.imag), axis=1)*2/var*np.sqrt(0.5)

def compensate(sym, cfo, baud):
    """Remove a carrier frequency offset (Hz) from symbols at `baud`"""
    return sym*np.exp(-2j*np.pi*cfo/baud*np.arange(len(sym)))

def decode_frame(z, index, phase, rate=None, n_pl=0):
    """Decode the PLFRAME whose SOF starts at z[index]. Needs config/coding/block ldpc/dvbs2_64800_<rate> and bch/dvbs2_<rate>

    z: symbols with the carrier offset removed except for a slow residual. The carrier phase over the frame is a line
    through the SOF and the pilot blocks (data-aided, so no phase slips); without pilots it is the SOF phase plus
    `rate` (radians per symbol, from neighbouring frames).
    Returns a dict: modcod info, whether LDPC converged, BCH corrections, and the descrambled BBFRAME bits
    (None if it did not decode). QPSK and 8PSK only (16/32APSK headers are reported, not decoded).
    """
    from .. import coding, scram, mod as modulation
    modcod, short, pilots, err = read_header(z, index, phase)
    kind, crate = MODCOD[modcod]
    res = {"modcod": modcod, "modulation": kind, "rate": crate, "short": short, "pilots": pilots, "pls_errors": err, "bbframe": None}
    if short or kind not in ("QPSK", "8PSK"):
        res["note"] = "not decoded: " + ("short frame" if short else kind)
        return res
    flen = frame_symbols(modcod, short, pilots)
    if index + flen > len(z):
        res["note"] = "frame runs past the end"
        return res
    d = pl_scramble(z[index + 90:index + flen], n_pl, True)
    ref = (1 + 1j)/np.sqrt(2)
    n = np.arange(len(d))
    blocks = pilot_blocks(modcod, short) if pilots else []
    if blocks:
        pos = np.array([-60.0] + [s + l/2 for s, l in blocks])
        ph = np.unwrap([phase] + [np.angle(np.mean(d[s:s+l]*np.conj(ref))) for s, l in blocks])
        ramp = np.polyval(np.polyfit(pos, ph, 1), n)
    else:
        ramp = phase + (rate or 0.0)*(n + 60)
    d = d*np.exp(-1j*ramp)
    var = float(np.mean(np.abs(np.concatenate([d[s:s+l] - ref for s, l in blocks]))**2)) if blocks else 0.1
    keep = np.ones(len(d), dtype=bool)
    for s, l in blocks:
        keep[s:s+l] = False
    data = d[keep]
    data = data/np.sqrt(np.mean(np.abs(data)**2))
    var = max(var, 1e-3)
    res["noise_var"] = var
    rf = _RATE_FILE[crate]
    try:
        ldpc = coding.load("block", f"ldpc/dvbs2_64800_{rf}")._enc
        bch = coding.load("block", f"bch/dvbs2_{rf}")._enc
    except FileNotFoundError as e:
        res["note"] = f"not decoded: missing config {e.filename.split('/')[-1]} (`python -m rpps gen-ldpc` makes the LDPC ones)"
        return res
    if kind == "8PSK":
        m = modulation.load("dvbs2_8psk")
        llr3 = demap_llr(data, m.points.arr, np.asarray(m.get_maps()[0].arr), var)
        cols = _COLS8.get(crate, (0, 1, 2))
        coded = np.concatenate([llr3[:, cols[c]] for c in range(3)])
    else: # QPSK: b0 -> I, b1 -> Q, no bit interleaver
        coded = (np.stack((data.real, data.imag), axis=1)*np.sqrt(2)*2/var).reshape(-1)
    cw = ldpc.decode_codewords(coded[None, :])
    res["ldpc_converged"] = bool(ldpc.converged[0])
    out = bch.decode(cw[0, :ldpc.num].astype(np.uint8))
    res["bch_corrected"] = int(bch.errors[0])
    if bch.errors[0] < 0:
        return res
    res["bbframe"] = scram.load("adt", "dvbs2_bb").decode(out.astype(np.uint8))
    res["header"] = parse_bbheader(res["bbframe"])
    return res

def decode(z, max_frames=None, cfo=None, baud=None):
    """Decode every PLFRAME found in symbols (pass cfo [Hz] and baud to remove the carrier offset first)"""
    if cfo is not None:
        z = compensate(z, cfo, baud)
    z = z/np.sqrt(np.mean(np.abs(z)**2))
    fr = find_frames(z)
    # Drift of the carrier between frames without pilots: from the SOF phases of neighbours
    rate = None
    if len(fr) >= 3:
        idx = np.array([f[0] for f in fr], dtype=float)
        ph = np.unwrap([f[1] for f in fr])
        rate = float(np.polyfit(idx, ph, 1)[0])
    out = []
    for i, p, _ in fr:
        out.append(decode_frame(z, i, p, rate))
        if max_frames and len(out) >= max_frames:
            break
    return out
