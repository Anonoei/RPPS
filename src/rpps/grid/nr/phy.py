"""5G NR downlink physical layer at 15 and 30 kHz subcarriers (FR1): SS/PBCH block, PBCH payload, CORESET 0, PDCCH, PDSCH

3GPP TS 38.211 (signals, mapping), 38.212 (7.1 BCH, 7.3 DCI), 38.213 (4.1: block positions; 13: CORESET 0 and Type0
search space), 38.214 (5.1 PDSCH). The numerology mu is 0 (15 kHz) or 1 (30 kHz): a slot is 14 symbols in 1 ms >> mu,
and at a sample rate of scs x nfft it is 15 nfft samples; the first symbol of every half millisecond has the longer
prefix. The SS/PBCH block and the channels it points to may differ in mu.

The grid of a slot is (14, nfft): every FFT bin, lowest frequency first, with the SS/PBCH block's subcarrier 120 at 0 Hz
(NrCell puts it there). So subcarrier k of the block is bin nfft/2 - 120 + k of a grid at its spacing, and every other
channel is placed from it: CORESET 0 from the block's first subcarrier, k_SSB and the table's offset.
"""
import functools

import numpy as np

from ..lte import phy as lte_phy

SPACINGS = {15e3: 0, 30e3: 1} # Subcarrier spacing [Hz] -> mu
gold, sign, ramp = lte_phy.gold, lte_phy.sign, lte_phy.ramp
SI_RNTI = 0xFFFF
# The first symbols of the candidate SS/PBCH blocks (38.213 4.1), counted from the half frame's start: the first ones,
# and the symbols they repeat after
SSB_PATTERNS = {"A": (0, (2, 8), 14), "B": (1, (4, 8, 16, 20), 28), "C": (1, (2, 8), 14)} # name -> (mu, symbols, period)
SI_CORESET_MAX_K_SSB = 23 # (a larger k_SSB: no CORESET 0 for this block)

def mu_of(scs):
    """The numerology of a subcarrier spacing [Hz]"""
    mu = SPACINGS.get(float(scs))
    if mu is None:
        raise ValueError(f"NR: subcarrier spacing {scs/1e3:g} kHz (15 or 30)")
    return mu

def cp_lengths(nfft, mu):
    """(the long prefix, the others') in samples"""
    cp = int(round(nfft*144/2048))
    return cp + int(round(nfft*(16 << mu)/2048)), cp

def slot_len(nfft):
    """Samples of a slot, at either spacing"""
    return 15*nfft

@functools.lru_cache(maxsize=32)
def symbol_starts(nfft, mu):
    """The useful part's first sample of a slot's 14 symbols"""
    long, cp = cp_lengths(nfft, mu)
    out, pos = [], 0
    for l in range(14):
        c = long if l % (14 if mu else 7) == 0 else cp
        out.append(pos + c)
        pos += c + nfft
    assert pos == slot_len(nfft)
    out = np.array(out)
    out.flags.writeable = False
    return out

def grid(x, nfft, mu, early=None):
    """The (14, nfft) grid of a slot's samples, lowest frequency first

    The window starts `early` samples into the prefix, its phase ramp taken off"""
    early = nfft//128 if early is None else early
    st = symbol_starts(nfft, mu) - early
    W = np.fft.fft(np.asarray(x)[st[:, None] + np.arange(nfft)[None, :]], axis=1)
    f = np.fft.fftfreq(nfft, 1/nfft)
    W = W*np.exp(2j*np.pi*f*early/nfft)[None, :]
    return np.fft.fftshift(W, axes=1)

def symbol_phase(fc, nfft, mu, slot=0):
    """Undoes the upconversion phase

    e^(j 2 pi fc t_l) (38.211 5.4) for a receiver whose 0 Hz is at fc (t_l: the useful part's start in the subframe; slot:
    the slot's number, which in the subframe is slot mod 2^mu)"""
    t = ((slot % (1 << mu))*slot_len(nfft) + symbol_starts(nfft, mu))/(15e3*(1 << mu)*nfft)
    return np.exp(2j*np.pi*((fc*t) % 1.0))

def modulate(G, nfft, mu):
    """Samples of slots from their grids (n*14, nfft), as `grid` reads them"""
    long, cp = cp_lengths(nfft, mu)
    out = []
    for i, row in enumerate(np.asarray(G)):
        t = np.fft.ifft(np.fft.ifftshift(row))*np.sqrt(nfft)
        c = long if i % (14 if mu else 7) == 0 else cp
        out.append(np.concatenate((t[-c:], t)))
    return np.concatenate(out)

# ---- synchronization signals (7.4.2)

def _m(taps, init, n=127):
    x = list(init)
    for i in range(n - 7):
        x.append(sum(x[i + t] for t in taps) % 2)
    return np.array(x)

_X_PSS = _m((4, 0), [0, 1, 1, 0, 1, 1, 1]) # x(i+7) = x(i+4) + x(i), [x(6)..x(0)] = 1110110
_X0 = _m((4, 0), [1, 0, 0, 0, 0, 0, 0])
_X1 = _m((1, 0), [1, 0, 0, 0, 0, 0, 0])

def pss(n_id2):
    n = np.arange(127)
    return 1 - 2.0*_X_PSS[(n + 43*n_id2) % 127]

def sss(n_id1, n_id2):
    n = np.arange(127)
    m0 = 15*(n_id1//112) + 5*n_id2
    m1 = n_id1 % 112
    return (1 - 2.0*_X0[(n + m0) % 127])*(1 - 2.0*_X1[(n + m1) % 127])

@functools.lru_cache(maxsize=None)
def sss_table(n_id2):
    return np.array([sss(n1, n_id2) for n1 in range(336)])

# ---- the SS/PBCH block (7.4.3.1)

def pbch_dmrs(pci, i_ssb_bar):
    """The 144 PBCH DM-RS symbols of block i_SSB-bar (i_SSB + 4 n_hf below 3 GHz)"""
    c = gold((1 << 11)*(i_ssb_bar + 1)*(pci//4 + 1) + (1 << 6)*(i_ssb_bar + 1) + pci % 4, 288)
    return (sign(c[0::2]) + 1j*sign(c[1::2]))/np.sqrt(2)

@functools.lru_cache(maxsize=1008)
def ssb_map(pci):
    """(k, l) of the block's PBCH (432) and of its DM-RS (144), in mapping order, k 0-239 and l 0-3 within the block"""
    v = pci % 4
    pk, pl, dk, dl = [], [], [], []
    for l in (1, 2, 3):
        ks = range(240) if l != 2 else [k for k in range(240) if k < 48 or k >= 192]
        for k in ks:
            if (k - v) % 4 == 0:
                dk.append(k); dl.append(l)
            else:
                pk.append(k); pl.append(l)
    return np.array(pk), np.array(pl), np.array(dk), np.array(dl)

def ssb_bins(nfft):
    """Bin of each of the block's 240 subcarriers"""
    return nfft//2 - 120 + np.arange(240)

def ssb_place(pattern, i_ssb):
    """(slot in the half frame, first symbol in it) of candidate block i_SSB"""
    _, first, period = SSB_PATTERNS[pattern]
    return divmod(first[i_ssb % len(first)] + period*(i_ssb//len(first)), 14)

def equalize(G, data, dmrs, ref):
    """(Y conj(H), |H|^2) of the data REs of a grid from the DM-RS

    data: (bins, symbols); dmrs: (bins, symbols), their values `ref`. Least squares at each, linear in frequency within a
    symbol, and linear in time between DM-RS symbols (constant past them)"""
    db, ds = data
    mb, ms = dmrs[0], dmrs[1]
    ls = G[ms, mb]/ref
    syms = np.unique(ms)
    Hs = np.array([np.interp(db, mb[ms == s], ls[ms == s].real) + 1j*np.interp(db, mb[ms == s], ls[ms == s].imag)
                   for s in syms]) # (DM-RS symbols, data REs)
    if len(syms) == 1:
        H = Hs[0]
    else:
        w = np.interp(ds, syms, np.arange(len(syms))) # Each data RE's place among the DM-RS symbols
        i0 = np.minimum(np.floor(w).astype(np.int64), len(syms) - 2)
        a = w - i0
        j = np.arange(len(db))
        H = Hs[i0, j]*(1 - a) + Hs[i0 + 1, j]*a
    return G[ds, db]*np.conj(H), np.abs(H)**2

@functools.lru_cache(maxsize=None)
def _pbch_code():
    from ... import code
    return code.load("block/polar/nr_pbch")

def _pbch_index(B, pci):
    """i_SSB-bar (0-7) of a block's grid, from its PBCH DM-RS"""
    pk, pl, dk, dl = ssb_map(pci)
    kk = np.arange(56, 183)
    h = B[0, kk]*pss(pci % 3) # (PSS values are +-1)
    H = np.interp(dk, kk, h.real) + 1j*np.interp(dk, kk, h.imag)
    r = B[dl, dk]*np.conj(H)
    # Per symbol, the magnitude: a residual offset turns each symbol's phase from the PSS's by its own amount
    score = [float(sum(np.abs(np.sum((r*np.conj(pbch_dmrs(pci, i)))[dl == l])) for l in (1, 2, 3))) for i in range(8)]
    return int(np.argmax(score))

def read_pbch(B, pci):
    """The PBCH of a block's grid (4, 240), or None where the CRC fails

    A dict (mib: its 24 bits, sfn, hrf, k_ssb_msb, i_ssb). The DM-RS says i_SSB-bar; the scrambling takes two bits of the
    block's index where a half frame has 4 candidates (i_SSB-bar then carries the half frame in its third) and three where it
    has more: both are tried, the CRC says which"""
    pk, pl, dk, dl = ssb_map(pci)
    ibar = _pbch_index(B, pci)
    z, _ = equalize(B, (pk, pl), (dk, dl), pbch_dmrs(pci, ibar))
    s = soft_bits(z, 2)
    for i_ssb in sorted({ibar & 3, ibar}):
        a, err = _pbch_code().decoder.decode(s*sign(pbch_scramble(pci, i_ssb)), status=True)
        if err[0] >= 0:
            mib, sfn, hrf, kmsb = payload_decode(np.asarray(a, np.uint8), pci)
            return {"mib": mib, "sfn": sfn, "hrf": hrf, "k_ssb_msb": kmsb, "i_ssb": i_ssb}
    return None

# ---- PBCH payload (38.212 7.1)

G_TABLE = [16, 23, 18, 17, 8, 30, 10, 6, 24, 7, 0, 5, 3, 2, 1, 4, 9, 11, 12, 13, 14, 15, 19, 20, 21, 22, 25, 26, 27, 28, 29, 31]
SFN_MIB = range(1, 7) # The SFN's six MSBs in the MIB (after its one-bit message choice)

def _payload_kinds():
    """For each of the 32 bits a_0..a_31 (MIB, then SFN LSBs 4th..1st, HRF, k_SSB MSB, two reserved): its kind"""
    return (["mib"] + ["sfn"]*6 + ["mib"]*17 + ["sfn"]*4 + ["hrf"] + ["ssb"]*3)

def _interleave_positions():
    """Where a_i goes in the interleaved payload: G(j) by kind (7.1.1)"""
    j = {"sfn": 0, "hrf": 10, "ssb": 11, "mib": 14}
    out = []
    for kind in _payload_kinds():
        out.append(G_TABLE[j[kind]])
        j[kind] += 1
    return np.array(out)

POS = _interleave_positions()
# Bits never scrambled: the half frame, the SFN's 2nd and 3rd LSBs (a_26, a_25), and, below 3 GHz, none of the three last
UNSCRAMBLED = np.array(sorted([POS[24 + 4], POS[24 + 1], POS[24 + 2]]))

def payload_scramble(pci, sfn_lsb23):
    """The 32 bits XORed onto the interleaved payload (7.1.2): c(j + v M), M = 29, v the SFN's 3rd and 2nd LSBs"""
    s = np.zeros(32, np.uint8)
    c = gold(pci, 29*4)
    j = 0
    for i in range(32):
        if i in UNSCRAMBLED:
            continue
        s[i] = c[j + sfn_lsb23*29]
        j += 1
    return s

def payload_encode(mib, sfn, hrf, k_ssb, pci):
    """The 32 PBCH payload bits (before CRC): the 24 MIB bits (SFN's MSBs in them), the timing bits, interleaved, scrambled"""
    mib = np.asarray(mib, np.uint8).copy()
    mib[1:7] = [(sfn >> (9 - i)) & 1 for i in range(6)]
    a = np.r_[mib, [(sfn >> 3) & 1, (sfn >> 2) & 1, (sfn >> 1) & 1, sfn & 1, hrf, (k_ssb >> 4) & 1, 0, 0]].astype(np.uint8)
    out = np.zeros(32, np.uint8)
    out[POS] = a
    return out ^ payload_scramble(pci, (sfn >> 1) & 3)

def payload_decode(bits, pci):
    """(MIB bits 24, sfn, hrf, k_SSB's MSB) from the 32 payload bits after the CRC"""
    b = np.asarray(bits, np.uint8)
    v = (int(b[POS[25]]) << 1) | int(b[POS[26]]) # The SFN's 3rd and 2nd LSBs: sent as they are
    a = (b ^ payload_scramble(pci, v))[POS]
    sfn = 0
    for x in list(a[1:7]) + list(a[24:28]):
        sfn = (sfn << 1) | int(x)
    return a[:24], sfn, int(a[28]), int(a[29])

def parse_mib(mib):
    """The MIB's fields (38.331 6.2.2) by name (SFN's MSBs left to payload_decode)"""
    v = lambda a, n: int("".join(str(int(x)) for x in mib[a:a + n]), 2)
    return {"scs_common": v(7, 1), "k_ssb_lsb": v(8, 4), "dmrs_pos": 2 + v(12, 1), "coreset0": v(13, 4), "ss0": v(17, 4),
            "barred": v(21, 1), "intra_freq": v(22, 1)}

def build_mib(scs_common=0, k_ssb=0, dmrs_pos=2, coreset0=0, ss0=0, barred=1, intra_freq=1):
    b = lambda v, n: [(int(v) >> (n - 1 - i)) & 1 for i in range(n)]
    return np.array([0] + [0]*6 + b(scs_common, 1) + b(k_ssb & 15, 4) + b(dmrs_pos - 2, 1) + b(coreset0, 4) + b(ss0, 4)
                    + b(barred, 1) + b(intra_freq, 1) + [0], np.uint8)

def pbch_scramble(pci, v, n=864):
    """PBCH scrambling (38.211 7.3.3.1)

    c(i + v 864), v the block index's two LSBs where a half frame has 4 candidates, three where it has more"""
    return gold(pci, 864*8)[v*864:v*864 + n]

# ---- CORESET 0 and its PDCCH (38.211 7.3.2, 7.4.1.3; 38.213 13)

def table(name):
    from ... import _config
    t = _config.read(_config.find("table", name))
    return t["columns"], t["rows"]

@functools.lru_cache(maxsize=None)
def coreset0(index, scs=15e3, scs_common=15e3, min_bw=5):
    """CORESET 0 (38.213 13)

    For the SS/PBCH block's spacing, the CORESET's, and the band's minimum channel bandwidth [MHz] (5 or 10; 40: the tables
    of its own where the block is at 30 kHz)"""
    name = f"nr/coreset0_{int(scs/1e3)}_{int(scs_common/1e3)}" + ("_40" if min_bw >= 40 and scs == 30e3 else "")
    cols, rows = table(name)
    r = next((dict(zip(cols, r)) for r in rows if r[0] == index), None)
    if r is None:
        raise ValueError(f"CORESET 0 index {index}: reserved")
    return r

@functools.lru_cache(maxsize=None)
def type0(index):
    cols, rows = table("nr/type0_fr1")
    return dict(zip(cols, next(r for r in rows if r[0] == index)))

def coreset0_first_bin(nfft, k_ssb, offset, scs=15e3, scs_common=15e3):
    """CORESET 0's first bin, or None

    The common resource block holding the SS/PBCH block's subcarrier 0 starts k_SSB subcarriers of 15 kHz before it, the
    CORESET `offset` of its own blocks below that"""
    below = (120*scs + k_ssb*15e3)/scs_common
    if below != round(below):
        return None
    return nfft//2 - int(below) - 12*offset

def type0_slots(ss0, i_ssb, coreset_symbols, mu=0):
    """(slots n0, n0 + 1), frame parity, first symbol of Type0 occasions

    For block i_SSB, at the PDCCH's mu; n0 + 1 past the frame's last slot is the next frame's first"""
    t = type0(ss0)
    x = (t["o"] << mu) + int(np.floor(i_ssb*t["m"]))
    n = 10 << mu
    first = t["first"]
    first = (0 if i_ssb % 2 == 0 else coreset_symbols) if first == "alt" else int(first)
    return (x % n, x % n + 1), (x//n) % 2, first

@functools.lru_cache(maxsize=256)
def cce_regs(n_rb, symbols, pci, L=6, R=2):
    """For each CCE, the (symbol, RB) of its 6 REGs (interleaved mapping, bundles of L = 6, R = 2, shift = cell id)"""
    n_reg = n_rb*symbols
    nb = n_reg//L
    C = nb//R
    f = np.zeros(nb, np.int64)
    for c in range(C):
        for r in range(R):
            f[c*R + r] = (r*C + c + pci) % nb
    regs = [(i % symbols, i//symbols) for i in range(n_reg)] # Numbered time first
    out = []
    for j in range(n_reg//6):
        bundles = [f[6*j//L + t] for t in range(6//L)]
        out.append([regs[b*L + q] for b in bundles for q in range(L)])
    return out

def pdcch_dmrs(pci, slot, symbol, n):
    """The PDCCH DM-RS of a symbol: n values (3 a RB)"""
    c = gold(((1 << 17)*(14*slot + symbol + 1)*(2*pci + 1) + 2*pci) % (1 << 31), 2*n)
    return (sign(c[0::2]) + 1j*sign(c[1::2]))/np.sqrt(2)

def dci_10_si_size(n_rb):
    return int(np.ceil(np.log2(n_rb*(n_rb + 1)/2))) + 4 + 1 + 5 + 2 + 1 + 15

def parse_dci_10_si(bits, n_rb):
    """DCI format 1_0 with the SI-RNTI (38.212 7.3.1.2.1)"""
    p = 0
    def take(n):
        nonlocal p
        v = int("".join(str(int(x)) for x in bits[p:p + n]), 2)
        p += n
        return v
    return {"fdra": take(int(np.ceil(np.log2(n_rb*(n_rb + 1)/2)))), "tdra": take(4), "vrb": take(1), "mcs": take(5), "rv": take(2),
            "sii": take(1)}

def riv(first, count, n):
    return n*(count - 1) + first if count - 1 <= n//2 else n*(n - count + 1) + (n - 1 - first)

def riv_decode(v, n):
    L, S = v//n + 1, v % n
    if S + L > n:
        L, S = n - L + 2, n - 1 - S
    return S, L

@functools.lru_cache(maxsize=64)
def vrb_interleaver(n_rb, bundle=2):
    """VRB-to-PRB (38.211 7.3.1.6)

    The SI-RNTI's in CORESET 0: bundles of `bundle` blocks, written into two rows"""
    n = -(-n_rb//bundle)
    cols = n//2
    f = list(range(n))
    for c in range(cols):
        for r in range(2):
            if c*2 + r < n - 1:
                f[c*2 + r] = r*cols + c
    out = np.array([bundle*f[v//bundle] + v % bundle for v in range(n_rb)])
    out.flags.writeable = False
    return out

# ---- PDSCH (38.211 7.3.1, 7.4.1.1; 38.214 5.1)

@functools.lru_cache(maxsize=None)
def tdra(row, dmrs_pos):
    cols, rows = table("nr/pdsch_tdra_a")
    return dict(zip(cols, next(r for r in rows if r[0] == row and r[1] == dmrs_pos)))

def dmrs_symbols(mapping, s, l, dmrs_pos):
    """PDSCH DM-RS symbols of the slot: single symbol, dmrs-AdditionalPosition pos2 (Table 7.4.1.1.2-3)"""
    if mapping == "A":
        d = s + l
        add = {8: [7], 9: [7], 10: [6, 9], 11: [6, 9], 12: [6, 9], 13: [7, 11], 14: [7, 11]}.get(d, [])
        return [dmrs_pos] + add
    add = {5: [4], 6: [4], 7: [4], 8: [3, 6], 9: [4, 7], 10: [4, 7], 11: [4, 8], 12: [5, 9], 13: [5, 9]}.get(l, [])
    return [s] + [s + a for a in add]

def pdsch_dmrs(pci, slot, symbol, n):
    c = gold(((1 << 17)*(14*slot + symbol + 1)*(2*pci + 1) + 2*pci) % (1 << 31), 2*n)
    return (sign(c[0::2]) + 1j*sign(c[1::2]))/np.sqrt(2)

@functools.lru_cache(maxsize=None)
def mcs(index):
    cols, rows = table("nr/mcs_pdsch_1")
    r = dict(zip(cols, rows[index]))
    return r["qm"], r["rate"]/1024

@functools.lru_cache(maxsize=None)
def _tbs_list():
    return [r[1] for r in table("nr/tbs")[1]]

def tbs(n_re_prb, n_prb, qm, rate, layers=1):
    """Transport block size (38.214 5.1.3.2) from the REs a PRB carries in the slot"""
    n_re = min(156, n_re_prb)*n_prb
    ninfo = n_re*rate*qm*layers
    if ninfo <= 3824:
        n = max(3, int(np.floor(np.log2(ninfo))) - 6)
        nq = max(24, (1 << n)*int(np.floor(ninfo/(1 << n))))
        return next(t for t in _tbs_list() if t >= nq)
    n = int(np.floor(np.log2(ninfo - 24))) - 5
    nq = max(3840, (1 << n)*int(np.floor((ninfo - 24)/(1 << n) + 0.5)))
    if rate <= 1/4:
        C = int(np.ceil((nq + 24)/3816))
        return 8*C*int(np.ceil((nq + 24)/(8*C))) - 24
    if nq > 8424:
        C = int(np.ceil((nq + 24)/8424))
        return 8*C*int(np.ceil((nq + 24)/(8*C))) - 24
    return 8*int(np.ceil((nq + 24)/8)) - 24

def pdsch_res(c0bin, prbs, s, l, mapping, dmrs_pos, blocks=()):
    """PDSCH data and DM-RS elements

    (bins, symbols) and (bins, symbols, sequence index), on the resource blocks `prbs` (from CORESET 0's lowest, in the
    virtual blocks' order), symbols s..s+l-1: around the DM-RS symbols, and the REs of the SS/PBCH blocks in the slot
    (`blocks`: of each, (first symbol, the symbol past its last, first bin, the bin past its last))"""
    dm = dmrs_symbols(mapping, s, l, dmrs_pos)
    sc = (c0bin + 12*np.asarray(prbs)[:, None] + np.arange(12)).reshape(-1)
    bins, syms = [], []
    for sym in range(s, s + l):
        if sym in dm:
            continue
        keep = np.ones(len(sc), bool)
        for first, end, low, high in blocks:
            if first <= sym < end:
                keep &= ~((sc >= low) & (sc < high))
        bins.append(sc[keep]); syms.append(np.full(int(keep.sum()), sym))
    up = np.sort(sc)
    db, ds, di = [], [], []
    for sym in dm:
        b = up[(up - c0bin) % 2 == 0]
        db.append(b); ds.append(np.full(len(b), sym)); di.append((b - c0bin)//2)
    return (np.concatenate(bins), np.concatenate(syms)), (np.concatenate(db), np.concatenate(ds), np.concatenate(di))

@functools.lru_cache(maxsize=4096)
def pdcch_res(c0bin, n_rb, symbols, first_symbol, pci, cce, L):
    """(bins, symbols) of a candidate's data REs in mapping order, and of its DM-RS (bins, symbols, sequence index)"""
    regs = [r for j in range(cce, cce + L) for r in cce_regs(n_rb, symbols, pci)[j]]
    data, dmrs = [], []
    for sym, rb in regs:
        for k in range(12):
            b = c0bin + 12*rb + k
            if k % 4 == 1:
                dmrs.append((first_symbol + sym, b, 3*rb + k//4))
            else:
                data.append((first_symbol + sym, b))
    data.sort()
    dmrs.sort()
    d = np.array(data)
    m = np.array(dmrs)
    return (d[:, 1], d[:, 0]), (m[:, 1], m[:, 0], m[:, 2])

@functools.lru_cache(maxsize=256)
def ldpc_for(tbs_bits, rate, E, rv, qm):
    """The code block's LDPC code for a transport block of TBS bits (one code block: K' = TBS + its CRC)"""
    from ... import code
    from ...code.types.nr_ldpc import base_graph, segment
    crc = "crc16_xmodem" if tbs_bits <= 3824 else "crc24_lte_a"
    B = tbs_bits + (16 if tbs_bits <= 3824 else 24)
    bg = base_graph(tbs_bits, rate)
    C, kp, K, Z = segment(B, bg)
    if C != 1:
        raise ValueError(f"NR PDSCH: {C} code blocks (one is what is covered)")
    return crc, code.load(f"block/ldpc/nr_bg{bg}", kp=kp, e=E, rv=rv, qm=qm, z=Z)

# ---- modulation (38.211 5.1)

def qam(bits, qm):
    """Symbols of bits, Qm 2, 4, 6 (Gray, 38.211 5.1.3-5.1.5)"""
    b = 1 - 2.0*np.asarray(bits, float).reshape(-1, qm)
    if qm == 2:
        return (b[:, 0] + 1j*b[:, 1])/np.sqrt(2)
    if qm == 4:
        return (b[:, 0]*(2 - b[:, 2]) + 1j*b[:, 1]*(2 - b[:, 3]))/np.sqrt(10)
    return (b[:, 0]*(4 - b[:, 2]*(2 - b[:, 4])) + 1j*b[:, 1]*(4 - b[:, 3]*(2 - b[:, 5])))/np.sqrt(42)

def soft_bits(z, qm, noise=1.0):
    """Max-log soft bits (positive = 0) of equalized symbols z (unit channel), Qm 2, 4, 6"""
    z = np.asarray(z)
    if qm == 2:
        s = np.empty(2*len(z))
        s[0::2], s[1::2] = z.real, z.imag
        return s*2*np.sqrt(2)/noise
    pts = qam(np.array([[(v >> (qm - 1 - i)) & 1 for i in range(qm)] for v in range(1 << qm)]).reshape(-1), qm)
    labels = np.array([[(v >> (qm - 1 - i)) & 1 for i in range(qm)] for v in range(1 << qm)])
    d = -np.abs(z[:, None] - pts[None, :])**2/noise
    out = np.empty((len(z), qm))
    for i in range(qm):
        out[:, i] = d[:, labels[:, i] == 0].max(axis=1) - d[:, labels[:, i] == 1].max(axis=1)
    return out.reshape(-1)
