"""802.11a/g OFDM receiver (6-54 Mbit/s): bursts in a 20 MS/s capture -> MAC frames

    python projects/wifi/wifi_ofdm.py data/wifi_ch11.cf64 --fs 20

IEEE 802.11-2020 clause 17. 20 MHz channel, 64-point FFT at 20 MS/s (312.5 kHz spacing), 16-sample cyclic prefix, 52 used subcarriers (48 data, 4 pilots at -21 -7 +7 +21).
Frame: short training field STF (10 x 0.8 us), long training field LTF (32-sample guard + 2 x 64), SIGNAL (one BPSK 1/2 symbol: rate 4, reserved 1, length 12,
parity 1, tail 6), then DATA symbols (16 SERVICE bits + PSDU + 6 tail + pad, scrambled by z^-7 + z^-4 + 1, convolutional code 133/171 with puncturing, interleaved, mapped
BPSK/QPSK/16-QAM/64-QAM). Beacons are sent as such legacy frames even by 802.11n/ac/ax access points, typically at 6 Mbit/s.
Statistic per step: STF delay-16 autocorrelation (1 for a real STF, ~0.1 noise); LTF correlation peak pair 64 samples apart; SIGNAL parity and tail bits (chance 1/128);
then the frame's CRC-32 (chance 2e-10).
"""
import argparse
import zlib

import numpy as np

import rpps as rp

FS = 20e6
LTF_NEG = [1, 1, -1, -1, 1, 1, -1, 1, -1, 1, 1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1, 1, 1, 1] # subcarriers -26..-1
LTF_POS = [1, -1, -1, 1, 1, -1, 1, -1, 1, -1, -1, -1, -1, -1, 1, 1, -1, -1, 1, -1, 1, -1, 1, 1, 1, 1] # +1..+26
LTF = np.array(LTF_NEG + [0] + LTF_POS, float) # index k + 26
USED = np.array([k for k in range(-26, 27) if k != 0])
PILOTS = np.array([-21, -7, 7, 21])
PILOT_VAL = np.array([1, 1, 1, -1], float)
DATA_K = np.array([k for k in USED if k not in PILOTS])

def _scrambler_seq(n, init=None):
    """x[n] = x[n-7] ^ x[n-4] from a seven-bit seed (x[-7..-1], default all ones): rpps scram/adt/wifi_ofdm"""
    seed = 127 if init is None else sum(int(b) << (6 - i) for i, b in enumerate(init))
    return rp.scram.load("wifi_ofdm", seed=seed).sequence(n)
PILOT_POL = 1 - 2.0*_scrambler_seq(127)

def bin_of(k):
    return np.asarray(k) % 64

LTF_TIME = np.fft.ifft(np.concatenate((np.zeros(0), np.where(np.arange(64) < 27, 0, 0))).astype(complex)) if False else None
def _ltf_time():
    X = np.zeros(64, complex)
    X[bin_of(USED)] = LTF[USED + 26]
    return np.fft.ifft(X)*np.sqrt(64) # one 64-sample LTF symbol
LTF_SYM = _ltf_time()

# --- rates: Mbit/s -> (bits per subcarrier, code rate, SIGNAL RATE field R1..R4)
RATES = {6: (1, (1, 2), "1101"), 9: (1, (3, 4), "1111"), 12: (2, (1, 2), "0101"), 18: (2, (3, 4), "0111"),
         24: (4, (1, 2), "1001"), 36: (4, (3, 4), "1011"), 48: (6, (2, 3), "0001"), 54: (6, (3, 4), "0011")}
RATE_OF = {v[2]: k for k, v in RATES.items()}
PUNCT = {(1, 2): [1, 1], (2, 3): [1, 1, 1, 0], (3, 4): [1, 1, 1, 0, 0, 1]} # keep-mask over the A0 B0 A1 B1 ... stream

# --- convolutional code (rate 1/2, generators 133 and 171 octal, terminated with six zero bits)
_G = (0o133, 0o171)

_CONV = rp.coding.Conv(_G, 7, termination="zero")

def viterbi_r12(llr_a, llr_b):
    """Soft Viterbi for the terminated K=7 code (rpps): llr_* positive = bit 0, erased (punctured) positions 0. Returns the input bits (tail included: the zero tail of a path that ends in state 0)"""
    bits = _CONV._dec.decode(np.column_stack((llr_a, llr_b)).reshape(-1)).astype(np.uint8)
    return np.concatenate((bits, np.zeros(6, np.uint8)))

def depuncture(llr, rate):
    """Soft bits in transmission order -> (A, B) streams with zeros where the code was punctured"""
    keep = np.array(PUNCT[rate])
    n_full = int(np.ceil(len(llr)/keep.sum()))*len(keep)
    full = np.zeros(n_full)
    idx = np.where(np.tile(keep, n_full//len(keep)) == 1)[0][:len(llr)]
    full[idx] = llr
    return full[0::2], full[1::2]

def interleave_perm(ncbps, nbpsc):
    """For each coded bit index k of the transmitted symbol: its position j after the two permutations of 17.3.5.7 (rpps interleave/block/wifi_ofdm)"""
    return np.argsort(rp.coding.load("interleave", "wifi_ofdm", ncbps=ncbps, nbpsc=nbpsc)._enc.src)

def demap(z, nbpsc):
    """Soft bits (positive = 0) from equalized symbols, in the order the mapper took the bits, per 802.11 Gray mapping, max-log"""
    if nbpsc == 1:
        return -z.real
    if nbpsc == 2:
        out = np.empty(2*len(z)); out[0::2] = -z.real; out[1::2] = -z.imag
        return out*np.sqrt(2)
    if nbpsc == 4:
        y = np.stack((z.real, z.imag))*np.sqrt(10)
        b = np.stack((-y, np.abs(y) - 2), -1) # (2, n, 2): [axis I/Q][symbol][bit]
        return np.stack((b[0], b[1]), -1).reshape(len(z), 4) .reshape(-1) if False else np.concatenate([np.stack((b[0][:, 0], b[0][:, 1], b[1][:, 0], b[1][:, 1]), 1)]).reshape(-1)
    y = np.stack((z.real, z.imag))*np.sqrt(42)
    bits = []
    for ax in (0, 1):
        v = y[ax]
        bits.append(np.stack((-v, np.abs(v) - 4, np.abs(np.abs(v) - 4) - 2), 1))
    return np.concatenate((bits[0], bits[1]), 1).reshape(-1)

def bytes_lsb(bits):
    n = len(bits)//8
    b = np.asarray(bits[:n*8], np.uint8).reshape(-1, 8)
    return bytes((b*(1 << np.arange(8))).sum(1).astype(np.uint8))

# --- receiver
def stf_metric(x, n0, n1, lag=16, win=128):
    """Delay-16 autocorrelation over n0..n1: returns (metric array, C) with metric = |C|/P"""
    seg = x[n0:n1 + win + lag]
    prod = seg[:-lag]*np.conj(seg[lag:])
    pw = np.abs(seg[lag:])**2
    cs = np.cumsum(np.r_[0, prod]); cp = np.cumsum(np.r_[0, pw])
    C = cs[win:win + (n1 - n0)] - cs[:(n1 - n0)]
    P = cp[win:win + (n1 - n0)] - cp[:(n1 - n0)]
    return np.abs(C)/np.maximum(P, 1e-30), C

def fft_sym(y, start):
    return np.fft.fft(y[start:start + 64])/np.sqrt(64)

def receive(x, s, e, fs=FS, sig_rot=1.0):
    """One burst -> dict(rate, length, frame, fcs_ok, ...) or None. x is the capture at 20 MS/s, s..e the burst."""
    # timing from the long training field: two copies of one known symbol 64 samples apart; a wide search (the STF plateau edge is not a reliable start)
    n0 = max(0, s - 60); n1 = min(len(x), s + 800)
    c = np.abs(np.correlate(x[n0:n1], LTF_SYM, mode="valid"))
    if len(c) < 140:
        return None
    pair = c[:-64] + c[64:]
    j = int(np.argmax(pair))
    if pair[j] < 12*np.median(c):
        return None
    base = n0 + j - 192 # the STF starts 160 + 32 (guard) samples before the first LTF symbol
    if base < 20 or base + 20000 > len(x) + 20000 or n0 + j + 128 + 80*3 > len(x):
        return None
    stf = x[base + 16:base + 128]
    acc = np.sum(stf[:-16]*np.conj(stf[16:]))
    m = np.abs(acc)/max(np.sum(np.abs(stf[16:])**2), 1e-30)
    if m < 0.5: # no short training field where the long one says it should be
        return None
    cfo = -np.angle(acc)*fs/(2*np.pi*16)
    seg = x[base - 20:base + 20000]
    seg = seg*np.exp(-2j*np.pi*cfo*np.arange(len(seg))/fs)
    start = base
    l1 = 20 + 192
    if l1 + 128 + 80*3 > len(seg):
        return None
    f1 = seg[l1:l1 + 64]; f2 = seg[l1 + 64:l1 + 128]
    fine = np.angle(np.sum(np.conj(f1)*f2))*fs/(2*np.pi*64)
    tt = np.arange(len(seg))
    seg = seg*np.exp(-2j*np.pi*fine*tt/fs)
    # channel estimate (early window: start 4 samples before the symbol: inside the guard interval, compensated below)
    early = 4
    def sym(start):
        w = np.fft.fft(seg[start - early:start - early + 64])/np.sqrt(64)
        return w[bin_of(USED)]*np.exp(2j*np.pi*USED*early/64)
    Y1 = sym(l1); Y2 = sym(l1 + 64)
    H = (Y1 + Y2)/2/LTF[USED + 26]
    Hd = dict(zip(USED, H))
    sig_start = l1 + 128 + 16 # useful part of the SIGNAL symbol (its 16-sample cyclic prefix first)
    def eq_symbol(n):
        st = l1 + 128 + 80*n + 16
        if st - early + 64 > len(seg):
            return None
        Y = sym(st)
        Z = Y/H
        # pilots: common phase and timing slope
        pk = np.array([list(USED).index(k) for k in PILOTS])
        pil = Z[pk]/(PILOT_VAL*PILOT_POL[n % 127])
        ph = np.unwrap(np.angle(pil))
        a, b = np.polyfit(PILOTS, ph, 1)
        corr = np.exp(-1j*(a*USED + b))
        Z = Z*corr
        d = np.array([list(USED).index(k) for k in DATA_K])
        return Z[d]
    z0 = eq_symbol(0)
    if z0 is None:
        return None
    z0 = z0*sig_rot # 1: the BPSK of a legacy SIGNAL; -1j would read a QBPSK symbol (HT-SIG)
    # SIGNAL: BPSK, rate 1/2, 24 bits
    perm = interleave_perm(48, 1)
    s_llr = np.empty(48); s_llr[perm] = -z0.real # received bit at position j is the coded bit k
    inv = np.empty(48); inv[:] = s_llr[perm]
    coded = np.zeros(48)
    coded[np.arange(48)] = s_llr[perm]
    # transmitted order: bit k went to position perm[k]; so soft bit k = value at position perm[k]
    soft = (-z0.real)[perm]
    a_, b_ = depuncture(soft, (1, 2))
    bits = viterbi_r12(a_, b_)
    if bits[18:24].any() or (int(bits[:18].sum()) & 1): # tail = bits 18-23 zero; the parity bit (17) makes bits 0-17 sum to even
        return None
    rate_key = "".join(str(int(v)) for v in bits[:4])
    if rate_key not in RATE_OF:
        return None
    rate = RATE_OF[rate_key]
    length = int(sum(int(bits[5 + i]) << i for i in range(12)))
    if not 14 <= length <= 4095:
        return None
    nbpsc, cr, _ = RATES[rate]
    ncbps = 48*nbpsc; ndbps = ncbps*cr[0]//cr[1]
    nsym = int(np.ceil((16 + 8*length + 6)/ndbps))
    if (e - s) < 0.7*(400 + 80*nsym): # SIGNAL parity and tail pass by chance 1 in 128: the burst must also be as long as the frame it announces
        return None
    perm = interleave_perm(ncbps, nbpsc)
    soft_all = []
    for n in range(1, nsym + 1):
        z = eq_symbol(n)
        if z is None:
            return dict(rate=rate, length=length, short=True, start=start)
        sb = demap(z, nbpsc)
        soft_all.append(sb[perm])
    soft_all = np.concatenate(soft_all)
    a_, b_ = depuncture(soft_all, cr)
    n_in = nsym*ndbps
    a_, b_ = a_[:n_in], b_[:n_in]
    dec = viterbi_r12(a_, b_)
    # scrambler: the first 7 SERVICE bits are zero, so the decoded bits are the scrambler's first 7 outputs
    seq = np.empty(n_in, np.uint8); seq[:7] = dec[:7]
    for i in range(7, n_in):
        seq[i] = seq[i - 7] ^ seq[i - 4]
    data = dec ^ seq
    psdu = bytes_lsb(data[16:16 + 8*length])
    ok = len(psdu) == length and zlib.crc32(psdu[:-4]) == int.from_bytes(psdu[-4:], "little")
    return dict(rate=rate, length=length, frame=psdu, fcs_ok=ok, cfo=cfo + fine, start=start, nsym=nsym, stf=float(m))

def bursts(x, fs=FS, factor=6.0, win=100, min_us=30):
    p = np.abs(x)**2
    e = p[:len(p)//win*win].reshape(-1, win).mean(1)
    on = e > factor*np.percentile(e, 20)
    d = np.diff(np.r_[0, on.astype(int), 0])
    st = np.where(d == 1)[0]*win; en = np.where(d == -1)[0]*win
    return [(s, e_) for s, e_ in zip(st, en) if (e_ - s)/fs*1e6 >= min_us]

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("path"); ap.add_argument("--fs", type=float, default=20)
    a = ap.parse_args(argv)
    x = np.fromfile(a.path, np.complex64).astype(np.complex128)
    bs = bursts(x)
    out = [r for r in (receive(x, s, e) for s, e in bs) if r]
    ok = [r for r in out if r.get("fcs_ok")]
    print(f"{len(bs)} bursts; SIGNAL fields valid: {len(out)}; frames with a valid FCS: {len(ok)}")
    return out

if __name__ == "__main__":
    main()
