"""LTE downlink physical layer: numerology, reference signals, resource-element maps, transmit diversity (3GPP TS 36.211)

FDD, normal cyclic prefix, 15 kHz subcarriers. A subframe (1 ms) is two slots of seven OFDM symbols; the resource grid of
a subframe is (14, 12 N_RB): symbol l = 0..13 (slot l // 7), subcarrier k = 0..12 N_RB - 1 from the lowest, the DC
subcarrier left out. Sample rates are 1.92 MS/s per FFT of 128: FFT = Fs / 15 kHz.

Maps give resource elements as two arrays (k, l) in the order the standard fills them, so a channel's symbols are
`G[l, k]`. Transmit diversity (6.3.4.3) is undone over those: 1 port as is, 2 ports Alamouti over pairs (SFBC), 4 ports
SFBC with frequency-switched pairs (ports 0, 2 then 1, 3).
"""
import functools

import numpy as np

SCS = 15e3
N_RB = {128: 6, 256: 15, 512: 25, 1024: 50, 1536: 75, 2048: 100} # FFT size -> resource blocks (the bandwidth that fills it)
CRS_SYMBOLS = {0: (0, 4), 1: (0, 4), 2: (1,), 3: (1,)} # Symbols of a slot that carry each port's reference signals
BANDWIDTH = {0: 6, 1: 15, 2: 25, 3: 50, 4: 75, 5: 100} # MIB dl-Bandwidth
PHICH_NG = (1/6, 1/2, 1.0, 2.0) # MIB phich-Resource
SI_RNTI, P_RNTI = 0xFFFF, 0xFFFE

# ---- numerology (6.12)

def cp_lengths(nfft):
    """(first symbol's prefix, the others') in samples at this FFT size"""
    s = nfft/2048
    return int(round(160*s)), int(round(144*s))

def subframe_len(nfft):
    cp0, cp = cp_lengths(nfft)
    return 2*(cp0 + nfft + 6*(cp + nfft))

def symbol_starts(nfft):
    """The useful part's first sample of the 14 symbols of a subframe, from the subframe's first sample"""
    cp0, cp = cp_lengths(nfft)
    slot = cp0 + nfft + 6*(cp + nfft)
    out = []
    for ns in (0, 1):
        for l in range(7):
            out.append(ns*slot + (cp0 if l == 0 else cp0 + nfft + (l - 1)*(cp + nfft) + cp))
    return np.array(out)

def sc_bin(k, nfft, n_rb):
    """FFT bin of subcarrier k = 0..12 N_RB - 1 (the DC bin skipped)"""
    k = np.asarray(k)
    return np.where(k < 6*n_rb, k - 6*n_rb, k - 6*n_rb + 1) % nfft

def grid(x, nfft, n_rb, early=None):
    """The (14, 12 N_RB) grid of one subframe's samples. Each FFT window starts `early` samples into the prefix (clear of
    the symbol before, and of a timing that runs late), the phase ramp that leaves taken off"""
    early = nfft//128 if early is None else early
    idx, bins, turn = _grid_plan(int(nfft), int(n_rb), int(early))
    W = np.fft.fft(np.asarray(x)[idx], axis=1)
    return W[:, bins]*turn[None, :]

@functools.lru_cache(maxsize=64)
def _grid_plan(nfft, n_rb, early):
    """What `grid` reads for a width: each symbol's FFT window, the subcarriers' bins, the ramp the early start leaves"""
    st = symbol_starts(nfft) - early
    idx = st[:, None] + np.arange(nfft)[None, :]
    k = np.arange(12*n_rb)
    f = np.where(k < 6*n_rb, k - 6*n_rb, k - 6*n_rb + 1)
    out = (idx, sc_bin(k, nfft, n_rb), np.exp(2j*np.pi*f*early/nfft))
    for a in out:
        a.flags.writeable = False
    return out

def ramp(n, start, step):
    """e^(-j 2 pi (start + step k)) for k = 0..n-1 (cycles): a derotation, complex64. Built from two short tables and their
    outer product (k = 128 a + b), not n exponentials: the phase is reduced to a cycle in float64 first, so it stays exact
    over a stream of any length"""
    b = np.arange(128)
    a = np.arange(-(-n//128))
    lo = np.exp(-2j*np.pi*((step*b) % 1.0))
    hi = np.exp(-2j*np.pi*((start + step*128*a) % 1.0))
    return (hi[:, None]*lo[None, :]).reshape(-1)[:n].astype(np.complex64)

# ---- sequences (7.2, 6.10.1)

@functools.lru_cache(maxsize=4096)
def _gold(c_init, n):
    from ... import sequence
    out = sequence.load("lte_gold", c_init=int(c_init)).bits(n).astype(np.uint8)
    out.flags.writeable = False
    return out

def gold(c_init, n):
    """c(0..n-1) of 7.2 (the rpps definition sequence/recurrence/lte_gold); lengths are taken in steps, so a channel's
    sequence of any length reuses one made before"""
    m = max(1024, 1 << int(np.ceil(np.log2(max(n, 1)))))
    return _gold(int(c_init), m)[:n]

def sign(c):
    """Bits to the signs a soft bit is multiplied by to descramble it: 0 -> +1, 1 -> -1"""
    return 1.0 - 2.0*np.asarray(c, float)

@functools.lru_cache(maxsize=4096)
def crs_values(pci, ns, l, n_rb):
    """r(m) for m = 0..2 N_RB - 1, of cell `pci`, slot ns (0..19), symbol l of the slot (normal prefix)"""
    c = gold((1 << 10)*(7*(ns + 1) + l + 1)*(2*pci + 1) + 2*pci + 1, 4*110)
    r = (sign(c[0::2]) + 1j*sign(c[1::2]))/np.sqrt(2)
    return r[np.arange(2*n_rb) + 110 - n_rb]

def crs_shift(port, ns, l):
    """v of 6.10.1.2, before v_shift = PCI mod 6"""
    if port == 0:
        return 0 if l == 0 else 3
    if port == 1:
        return 3 if l == 0 else 0
    if port == 2:
        return 3*(ns % 2)
    return 3 + 3*(ns % 2)

def crs_mask(pci, n_rb, ports):
    """(14, 12 N_RB) bool: where the reference signals of ports 0..ports-1 are (no data there)"""
    m = np.zeros((14, 12*n_rb), bool)
    for p in range(ports):
        for slot in (0, 1):
            for l in CRS_SYMBOLS[p]:
                m[slot*7 + l, (crs_shift(p, slot, l) + pci) % 6::6] = True
    return m

def channel(G, pci, n_rb, sf, ports=4):
    """H (4, 14, 12 N_RB): each port's channel on every resource element of subframe sf's grid G. Least squares at the
    reference signals, linear in frequency between them (constant past the edges), and a straight line in time through
    the port's symbols of the subframe (it barely moves in 1 ms). Ports past `ports` are zeros. Every port and symbol at
    once: a port's reference signals are a comb of step 6, so each subcarrier's two neighbours are found by arithmetic"""
    nsc = 12*n_rb
    H = np.zeros((4, 14, nsc), np.complex64)
    plan = _channel_plan(int(pci), int(n_rb), int(sf), int(ports))
    if plan is None:
        return H
    sym, kk, ref, i0, a, r, per_port = plan
    h = G[sym[:, None], kk]/ref # Least squares at each reference signal
    hs = h[r, i0]*(1 - a) + h[r, i0 + 1]*a # (rows, nsc)
    for p, (sel, dt, den, tt) in enumerate(per_port):
        H_p = hs[sel]
        mean = H_p.mean(0)
        slope = np.sum(dt[:, None]*(H_p - mean), axis=0)/den
        H[p] = mean[None, :] + slope[None, :]*tt[:, None]
    return H

@functools.lru_cache(maxsize=512)
def _channel_plan(pci, n_rb, sf, ports):
    """What `channel` reads of a subframe's reference signals, which only the cell, its width, the subframe and the ports
    decide: the rows (port, symbol), their subcarriers and values, each subcarrier's neighbours on its row's comb, and
    each port's line in time"""
    nsc = 12*n_rb
    rows = [(p, slot, l) for p in range(ports) for slot in (0, 1) for l in CRS_SYMBOLS[p]]
    if not rows:
        return None
    v = np.array([(crs_shift(p, 2*sf + slot, l) + pci) % 6 for p, slot, l in rows])
    sym = np.array([slot*7 + l for _, slot, l in rows])
    kk = 6*np.arange(2*n_rb)[None, :] + v[:, None] # (rows, 2 N_RB)
    ref = np.array([crs_values(pci, 2*sf + slot, l, n_rb) for p, slot, l in rows])
    t = np.clip((np.arange(nsc)[None, :] - v[:, None])/6.0, 0, 2*n_rb - 1) # Where each subcarrier falls on its row's comb
    i0 = np.minimum(np.floor(t).astype(np.int64), 2*n_rb - 2)
    a = t - i0
    r = np.arange(len(rows))[:, None]
    per_port = []
    for p in range(ports):
        sel = np.array([q == p for q, _, _ in rows])
        t_p = sym[sel].astype(float)
        tm = t_p.mean()
        per_port.append((sel, t_p - tm, np.sum((t_p - tm)**2), np.arange(14, dtype=float) - tm))
    out = (sym, kk, ref, i0, a, r, per_port)
    for x in out[:-1]:
        x.flags.writeable = False
    return out

# ---- transmit diversity (6.3.4.3), soft bits (7.1.2)

def undo_diversity(G, H, k, l, ports):
    """The symbols sent on resource elements (k, l), in that order, for 1, 2 or 4 transmit ports. Each is scaled by the
    channel's power (soft bits keep their reliability)"""
    r = G[l, k]
    h = H[:, l, k]
    z = np.zeros(len(k), complex)
    if ports == 1:
        return np.conj(h[0])*r
    n = len(k)//2*2
    if ports == 2:
        a, b = r[0:n:2], r[1:n:2]
        ha, hb = (h[0, 0:n:2] + h[0, 1:n:2])/2, (h[1, 0:n:2] + h[1, 1:n:2])/2 # A pair shares one channel
        z[0:n:2] = np.conj(ha)*a + hb*np.conj(b)
        z[1:n:2] = np.conj(ha)*b - hb*np.conj(a)
        return z
    n = len(k)//4*4
    for q, (p1, p2) in ((0, (0, 2)), (2, (1, 3))): # Quartets: the first pair on ports 0 and 2, the second on 1 and 3
        a, b = r[q:n:4], r[q + 1:n:4]
        ha = (h[p1, q:n:4] + h[p1, q + 1:n:4])/2
        hb = (h[p2, q:n:4] + h[p2, q + 1:n:4])/2
        z[q:n:4] = np.conj(ha)*a + hb*np.conj(b)
        z[q + 1:n:4] = np.conj(ha)*b - hb*np.conj(a)
    return z

def soft_bits(z):
    """QPSK (b(2i) on the real part, b(2i+1) on the imaginary, a 0 positive): soft bits, positive = 0"""
    s = np.empty(2*len(z))
    s[0::2], s[1::2] = z.real, z.imag
    return s

def qpskness(z):
    """-Re(sum z^4) / sum |z|^4: 1 for QPSK on the diagonals, ~0 for noise or the wrong equalizer"""
    return float(-np.sum(z**4).real/max(np.sum(np.abs(z)**4), 1e-30))

# ---- PBCH (6.6)

@functools.lru_cache(maxsize=64)
def pbch_res(pci, n_rb):
    """(k, l) of the 240 PBCH resource elements of subframe 0: slot 1 symbols 0-3, the 72 central subcarriers, around the
    reference signals of four ports whatever the cell has"""
    k0 = 6*n_rb - 36
    ks, ls = [], []
    for l in range(4):
        for k in range(k0, k0 + 72):
            if l < 2 and (k - pci % 6) % 3 == 0:
                continue
            ks.append(k)
            ls.append(7 + l)
    return np.array(ks), np.array(ls)

def parse_mib(b):
    """The 24 MIB bits (36.331 6.2.2): dl-Bandwidth 3, phich-Config 3, the SFN's 8 MSBs, then 10 bits of later releases"""
    v = lambda a, n: int("".join(str(int(x)) for x in b[a:a + n]), 2)
    return {"n_rb": BANDWIDTH.get(v(0, 3)), "phich_duration": v(3, 1), "phich_ng": PHICH_NG[v(4, 2)], "sfn_msb": v(6, 8)}

# ---- control region: PCFICH (6.7), PHICH (6.9), PDCCH (6.8)

CFI_WORDS = {1: np.array([0, 1, 1]*10 + [0, 1]), 2: np.array([1, 0, 1]*10 + [1, 0]), 3: np.array([1, 1, 0]*10 + [1, 1])}

def _reg_step(l, ports):
    """A REG is 6 subcarriers where reference signals are (symbol 0; symbol 1 with 4 ports), else 4 (6.2.4)"""
    return 6 if l == 0 or (l == 1 and ports == 4) else 4

def reg_res(k0, l, pci, ports):
    """The four resource elements (k, l) of the REG from subcarrier k0"""
    if _reg_step(l, ports) == 6: # (around the reference signals of ports 0 and 1, even in a one-port cell)
        return [k for k in range(k0, k0 + 6) if (k - pci % 6) % 3 != 0]
    return list(range(k0, k0 + 4))

def _res(regs, pci, ports):
    ks, ls = [], []
    for k0, l in regs:
        kk = reg_res(k0, l, pci, ports)
        ks += kk
        ls += [l]*len(kk)
    return np.array(ks), np.array(ls)

def pcfich_regs(pci, n_rb):
    nsc, kbar = 12*n_rb, 6*(pci % (2*n_rb))
    return [((kbar + (i*n_rb//2)*6) % nsc, 0) for i in range(4)]

@functools.lru_cache(maxsize=256)
def pcfich_res(pci, n_rb, ports):
    return _res(pcfich_regs(pci, n_rb), pci, ports)

@functools.lru_cache(maxsize=256)
def control_regs(pci, n_rb, cfi, phich_groups, ports):
    """The PDCCH REGs (k0, l) of a subframe in the order they are filled (time first), past the PCFICH and PHICH (normal
    duration)"""
    nsc = 12*n_rb
    pcf = pcfich_regs(pci, n_rb)
    sym0 = [k for k in range(0, nsc, 6) if (k, 0) not in pcf]
    n0 = len(sym0)
    phich = {(sym0[(pci + m + (i*n0)//3) % n0], 0) for m in range(phich_groups) for i in range(3)}
    used = set(pcf) | phich
    return [(k, l) for k in range(nsc) for l in range(cfi) if k % _reg_step(l, ports) == 0 and (k, l) not in used]

def control_symbols(cfi, n_rb):
    """OFDM symbols of the control region: the CFI, one more at 10 resource blocks or fewer (Table 6.7-1)"""
    return int(cfi) + (1 if n_rb <= 10 else 0)

def phich_groups(ng, n_rb):
    return int(np.ceil(ng*n_rb/8))

@functools.lru_cache(maxsize=256)
def control_res(pci, n_rb, symbols, phich_groups_, ports):
    """(k, l) of the PDCCH REGs' resource elements in order (control_regs, each REG's four)"""
    return _res(control_regs(pci, n_rb, symbols, phich_groups_, ports), pci, ports)

@functools.lru_cache(maxsize=256)
def subblock_perm(n):
    """The 32-column sub-block interleaver's read-out order of n items (36.212 5.1.4.2.1, dummies left out)"""
    from ... import code
    return code.load("interleave/lte_subblock", length=n).encoder.src

# ---- DCI (36.212 5.3.3)

AMBIGUOUS = {12, 14, 16, 20, 24, 26, 32, 40, 44, 56}

def riv_bits(n_rb):
    return int(np.ceil(np.log2(n_rb*(n_rb + 1)/2)))

def dci_1a_size(n_rb):
    """Format 1A's payload bits (FDD), padded to format 0's and past the ambiguous sizes (5.3.3.1.3)"""
    f1a = 1 + 1 + riv_bits(n_rb) + 5 + 3 + 1 + 2 + 2
    f0 = 1 + 1 + riv_bits(n_rb) + 5 + 1 + 2 + 3 + 1
    n = max(f1a, f0)
    return n + 1 if n in AMBIGUOUS else n

def dci_1c_size(n_rb):
    """Format 1C's payload bits (5.3.3.1.4): gap flag (N_RB >= 50), the allocation in steps of N_RB^step, 5 bits of TBS"""
    step = 2 if n_rb < 50 else 4
    nvrb = (n_rb//step)
    gap = 1 if n_rb >= 50 else 0
    return gap + int(np.ceil(np.log2(nvrb*(nvrb + 1)/2))) + 5

def parse_1a(bits, n_rb):
    """DCI 1A with an SI-, P- or RA-RNTI: flag, localized/distributed, RIV, MCS, HARQ, NDI, RV, TPC"""
    p = 0
    def take(n):
        nonlocal p
        v = int("".join(str(int(x)) for x in bits[p:p + n]), 2)
        p += n
        return v
    return {"flag": take(1), "distributed": take(1), "riv": take(riv_bits(n_rb)), "mcs": take(5), "harq": take(3),
            "ndi": take(1), "rv": take(2), "tpc": take(2)}

def riv_decode(riv, n_rb):
    """(first resource block, count) of a contiguous allocation (36.213 7.1.6.3)"""
    L, S = riv//n_rb + 1, riv % n_rb
    if S + L > n_rb:
        L, S = n_rb - L + 2, n_rb - 1 - S
    return S, L

# ---- PDSCH (6.3.5, 6.4)

def pdsch_res(first, count, cfi, pci, sf, n_rb, ports):
    """(k, l) of a localized PDSCH on resource blocks first..first+count-1: subcarrier first, then symbol, from symbol
    `cfi`, around the reference signals, and in subframes 0 and 5 around the synchronization signals (and the PBCH in 0)"""
    ks = np.arange(12*first, 12*(first + count))
    mask = crs_mask(pci, n_rb, ports)
    mid = (6*n_rb - 36, 6*n_rb + 36)
    centre = (ks >= mid[0]) & (ks < mid[1])
    K, L = [], []
    for l in range(cfi, 14):
        keep = ~mask[l, ks]
        if (sf in (0, 5) and l in (5, 6)) or (sf == 0 and l in (7, 8, 9, 10)):
            keep &= ~centre
        K.append(ks[keep])
        L.append(np.full(int(keep.sum()), l))
    return np.concatenate(K), np.concatenate(L)
