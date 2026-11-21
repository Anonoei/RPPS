"""5G NR downlink physical layer at 15 kHz subcarriers (mu = 0): SS/PBCH block, PBCH payload, CORESET 0, PDCCH, PDSCH

3GPP TS 38.211 (signals, mapping), 38.212 (7.1 BCH, 7.3 DCI), 38.213 (13: CORESET 0 and Type0 search space), 38.214
(5.1 PDSCH). A slot is 14 symbols in 1 ms, the cyclic prefixes LTE's (rpps.grid.lte.phy: the same numerology at mu = 0).

The grid of a slot is (14, nfft): every FFT bin, lowest frequency first, with the SS/PBCH block's subcarrier 120 at bin
nfft/2 (NrCell puts it there). So subcarrier k of the SS/PBCH block is bin nfft/2 - 120 + k, and every other channel is
placed from it: CORESET 0 from the block's first resource block, k_SSB and the table's offset.
"""
import functools

import numpy as np

from ..lte import phy as lte_phy

SCS = 15e3
cp_lengths, symbol_starts, slot_len = lte_phy.cp_lengths, lte_phy.symbol_starts, lte_phy.subframe_len
gold, sign = lte_phy.gold, lte_phy.sign
SI_RNTI = 0xFFFF
SSB_SYMBOLS = (2, 8) # Case A: the first symbols of the two candidate blocks of a slot (38.213 4.1), blocks 0-3 in slots 0, 1
LMAX = 4 # Candidate blocks in a half frame below 3 GHz

def grid(x, nfft, early=None):
    """The (14, nfft) grid of a slot's samples, lowest frequency first (the window `early` samples into the prefix, its
    phase ramp taken off)"""
    early = nfft//128 if early is None else early
    st = symbol_starts(nfft) - early
    W = np.fft.fft(np.asarray(x)[st[:, None] + np.arange(nfft)[None, :]], axis=1)
    f = np.fft.fftfreq(nfft, 1/nfft)
    W = W*np.exp(2j*np.pi*f*early/nfft)[None, :]
    return np.fft.fftshift(W, axes=1)

def symbol_phase(fc, nfft):
    """e^(j 2 pi fc t_l) for the 14 symbols of a slot: what undoes the phase NR's upconversion gives symbol l (38.211 5.4)
    for a receiver whose 0 Hz is at fc (t_l: the useful part's start in the subframe; mu = 0, a slot is a subframe)"""
    t = symbol_starts(nfft)/(SCS*nfft)
    return np.exp(2j*np.pi*((fc*t) % 1.0))

def modulate(G, nfft):
    """Samples of slots from their grids (n*14, nfft), as `grid` reads them"""
    cp0, cp = cp_lengths(nfft)
    out = []
    for i, row in enumerate(np.asarray(G)):
        t = np.fft.ifft(np.fft.ifftshift(row))*np.sqrt(nfft)
        c = cp0 if i % 7 == 0 else cp
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

def ssb_place(pci, i_ssb):
    """(slot in the half frame, first symbol) of candidate block i_SSB"""
    return i_ssb//2, SSB_SYMBOLS[i_ssb % 2]

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

def pbch_scramble(pci, i_ssb, n=864):
    """The bits XORed onto the 864 coded PBCH bits of block i_SSB (38.211 7.3.3.1: c(i + v 864), v its 2 LSBs)"""
    return gold(pci, 864*4)[(i_ssb & 3)*864:(i_ssb & 3)*864 + n]

# ---- CORESET 0 and its PDCCH (38.211 7.3.2, 7.4.1.3; 38.213 13)

def table(name):
    from ... import _config
    t = _config.read(_config.find("table", name))
    return t["columns"], t["rows"]

@functools.lru_cache(maxsize=None)
def coreset0(index):
    cols, rows = table("nr/coreset0_15_15")
    r = next((dict(zip(cols, r)) for r in rows if r[0] == index), None)
    if r is None:
        raise ValueError(f"CORESET 0 index {index}: reserved")
    return r

@functools.lru_cache(maxsize=None)
def type0(index):
    cols, rows = table("nr/type0_fr1")
    return dict(zip(cols, next(r for r in rows if r[0] == index)))

def coreset0_first_bin(nfft, k_ssb, offset):
    """Bin of CORESET 0's subcarrier 0: the common RB holding the block's subcarrier 0 starts k_SSB before it, the
    CORESET `offset` RBs below that"""
    return nfft//2 - 120 - k_ssb - 12*offset

def type0_slots(ss0, i_ssb, coreset_symbols):
    """(slots n0 and n0 + 1 of the frame, the parity of the frames, first symbol) of block i_SSB's Type0 occasions"""
    t = type0(ss0)
    x = t["o"] + int(np.floor(i_ssb*t["m"]))
    n0 = x % 10
    first = t["first"]
    first = (0 if i_ssb % 2 == 0 else coreset_symbols) if first == "alt" else int(first)
    return (n0, n0 + 1), (x//10) % 2, first

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

def pdsch_res(nfft, c0bin, first, count, s, l, mapping, dmrs_pos, ssb_slots):
    """(bins, symbols) of the PDSCH's data REs in mapping order, and of its DM-RS (bins, symbols, sequence index): around
    the DM-RS symbols, and the SS/PBCH block's REs where it is in the slot (ssb_slots: its first symbols here)"""
    dm = dmrs_symbols(mapping, s, l, dmrs_pos)
    sc = c0bin + 12*first + np.arange(12*count) # Subcarriers of the allocation
    ssb = ssb_bins(nfft)
    bins, syms = [], []
    for sym in range(s, s + l):
        if sym in dm:
            continue
        keep = np.ones(len(sc), bool)
        if any(l0 <= sym < l0 + 4 for l0 in ssb_slots):
            keep &= ~((sc >= ssb[0]) & (sc <= ssb[-1]))
        bins.append(sc[keep]); syms.append(np.full(int(keep.sum()), sym))
    db, ds, di = [], [], []
    for sym in dm:
        b = sc[(sc - c0bin) % 2 == 0]
        db.append(b); ds.append(np.full(len(b), sym)); di.append((b - c0bin)//2)
    return (np.concatenate(bins), np.concatenate(syms)), (np.concatenate(db), np.concatenate(ds), np.concatenate(di))

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
