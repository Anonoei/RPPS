"""An LTE downlink to test the receive side with: cell signals, MIB, CFI, a DCI 1A to the SI-RNTI and its PDSCH

    x = rp.grid.lte.tx.downlink(pci=77, ports=2, n_rb=15, frames=8, sib=bits)    # 3.84 MS/s, from subframe 0 of SFN 0

Every subframe carries the reference signals of its ports and a PCFICH; subframes 0 and 5 the PSS and SSS; subframe 0
the PBCH (the MIB of its frame's SFN); subframe 5 of every even frame a DCI 1A in the common search space and the
transport block `sib` on the PDSCH it allocates (36.331 5.2.1.2: where SIB1 goes). Transmit diversity over the ports
(6.3.4.3), each port's resource grid to samples by its own IFFT; `channel` puts each port through a gain of its own, sums
them and adds a carrier offset, a delay and noise. FDD, normal cyclic prefix (3GPP TS 36.211, 36.212, 36.213).
"""
import numpy as np

from . import phy
from .cell import pss, sss
from .channels import _conv, _conv_rm, _turbo, _turbo_rm, crc_bits, tbs_table

def qpsk(bits):
    b = np.asarray(bits, float)
    return ((1 - 2*b[0::2]) + 1j*(1 - 2*b[1::2]))/np.sqrt(2)

def diversity(x, ports):
    """(ports, len(x)): what each port sends on the resource elements of a channel, in order (6.3.4.3)"""
    x = np.asarray(x, complex)
    y = np.zeros((ports, len(x)), complex)
    if ports == 1:
        y[0] = x
        return y
    s = 1/np.sqrt(2)
    if ports == 2:
        a, b = x[0::2], x[1::2]
        y[0, 0::2], y[0, 1::2] = s*a, s*b
        y[1, 0::2], y[1, 1::2] = -s*np.conj(b), s*np.conj(a)
        return y
    for q, (p1, p2) in ((0, (0, 2)), (2, (1, 3))):
        a, b = x[q::4], x[q + 1::4]
        y[p1, q::4], y[p1, q + 1::4] = s*a, s*b
        y[p2, q::4], y[p2, q + 1::4] = -s*np.conj(b), s*np.conj(a)
    return y

def riv(first, count, n_rb):
    """The resource indication value of a contiguous allocation (36.213 7.1.6.3)"""
    if count - 1 <= n_rb//2:
        return n_rb*(count - 1) + first
    return n_rb*(n_rb - count + 1) + (n_rb - 1 - first)

def _bits(v, n):
    return [(int(v) >> (n - 1 - i)) & 1 for i in range(n)]

def mib(n_rb, sfn, phich_ng=1.0, phich_duration=0):
    bw = {v: k for k, v in phy.BANDWIDTH.items()}[n_rb]
    return np.array(_bits(bw, 3) + _bits(phich_duration, 1) + _bits(phy.PHICH_NG.index(phich_ng), 2) + _bits(sfn >> 2, 8)
                    + [0]*10, np.uint8)

def tail_biting_encode(bits, E):
    c = _conv().encoder.encode(np.asarray(bits, np.uint8))
    return np.asarray(_conv_rm(len(bits), E).encode(c)).astype(np.uint8)

def dci_1a(n_rb, first, count, mcs, rv=0, tpc=0):
    """The DCI 1A payload bits for a localized allocation, padded to its size (36.212 5.3.3.1.3)"""
    b = [1, 0] + _bits(riv(first, count, n_rb), phy.riv_bits(n_rb)) + _bits(mcs, 5) + _bits(0, 3) + [0] + _bits(rv, 2) + _bits(tpc, 2)
    return np.array(b + [0]*(phy.dci_1a_size(n_rb) - len(b)), np.uint8)

def subframe(pci, sf, sfn, n_rb, ports, cfi=1, phich_ng=1.0, dci=None, sib=None, rv=0, rnti=phy.SI_RNTI, cce=0, L=4):
    """(ports, 14, 12 N_RB): the resource grid each port sends in subframe sf of frame sfn"""
    nsc = 12*n_rb
    X = np.zeros((ports, 14, nsc), complex)
    for p in range(ports): # Reference signals
        for slot in (0, 1):
            ns = 2*sf + slot
            for l in phy.CRS_SYMBOLS[p]:
                X[p, slot*7 + l, (phy.crs_shift(p, ns, l) + pci) % 6::6] = phy.crs_values(pci, ns, l, n_rb)
    if sf in (0, 5): # Synchronization signals, on port 0
        k = 6*n_rb - 31 + np.arange(62)
        X[0, 6, k] = pss((25, 29, 34)[pci % 3])
        X[0, 5, k] = sss(pci//3, pci % 3, sf // 5)
    def put(sym, k, l):
        X[:, l, k] += diversity(sym, ports)
    if sf == 0: # PBCH
        m = mib(n_rb, sfn, phich_ng)
        crc = crc_bits(m, "crc16_xmodem") ^ np.array({1: [0]*16, 2: [1]*16, 4: [0, 1]*8}[ports], np.uint8)
        e = tail_biting_encode(np.concatenate((m, crc)), 1920) ^ phy.gold(pci, 1920)
        q = sfn % 4
        put(qpsk(e[q*480:(q + 1)*480]), *phy.pbch_res(pci, n_rb))
    w = phy.CFI_WORDS[cfi] ^ phy.gold(((sf + 1)*(2*pci + 1) << 9) + pci, 32) # PCFICH
    put(qpsk(w), *phy.pcfich_res(pci, n_rb, ports))
    if dci is not None: # PDCCH: the DCI in CCEs cce..cce+L-1, the rest of the region <NIL>
        regs = phy.control_regs(pci, n_rb, phy.control_symbols(cfi, n_rb), phy.phich_groups(phich_ng, n_rb), ports)
        M = len(regs)
        K = len(dci)
        c = np.concatenate((dci, crc_bits(dci, "crc16_xmodem") ^ np.array(_bits(rnti, 16), np.uint8)))
        bits = np.zeros(8*M, np.uint8)
        bits[72*cce:72*(cce + L)] = tail_biting_encode(c, 72*L)
        nil = np.ones(8*M, bool)
        nil[72*cce:72*(cce + L)] = False
        bits ^= phy.gold((sf << 9) + pci, 8*M)
        quads = qpsk(bits).reshape(M, 4)
        quads[nil.reshape(M, 8)[:, 0]] = 0
        w = quads[np.asarray(phy.subblock_perm(M))] # The quadruplet interleaver, then the shift by the cell
        wbar = w[(np.arange(M) + pci) % M]
        put(wbar.reshape(-1), *phy._res(regs, pci, ports))
        del K
    if sib is not None: # PDSCH: the transport block the DCI allocated
        p = phy.parse_1a(dci, n_rb)
        first, count = phy.riv_decode(p["riv"], n_rb)
        k, l = phy.pdsch_res(first, count, phy.control_symbols(cfi, n_rb), pci, sf, n_rb, ports)
        tb = np.asarray(sib, np.uint8)
        c = np.concatenate((tb, crc_bits(tb, "crc24_lte_a")))
        K = len(c)
        d = np.asarray(_turbo(K).encode(c)).astype(np.uint8)
        e = np.asarray(_turbo_rm(K, 2*len(k), rv).encode(d)).astype(np.uint8)
        e ^= phy.gold((rnti << 14) + (sf << 9) + pci, len(e))
        put(qpsk(e), k, l)
    return X

def modulate(X, nfft):
    """(ports, samples): each port's grid (ports, 14 n, 12 N_RB) to samples, a subframe per 14 symbols"""
    ports, nsym, nsc = X.shape
    n_rb = nsc//12
    cp0, cp = phy.cp_lengths(nfft)
    b = phy.sc_bin(np.arange(nsc), nfft, n_rb)
    out = []
    for i in range(nsym):
        F = np.zeros((ports, nfft), complex)
        F[:, b] = X[:, i]
        t = np.fft.ifft(F, axis=1)*np.sqrt(nfft)
        c = cp0 if i % 7 == 0 else cp
        out.append(np.concatenate((t[:, -c:], t), axis=1))
    return np.concatenate(out, axis=1)

def downlink(pci=77, ports=2, n_rb=15, frames=4, sib=None, mcs=None, first=None, cfi=None, phich_ng=1.0, sfn0=0, L=4):
    """(ports, samples) at the rate of n_rb's FFT (6 RB: 1.92 MS/s, 15: 3.84, 25: 7.68, 50: 15.36), from subframe 0 of
    frame sfn0, `frames` radio frames. sib: the transport block in subframe 5 of every even frame (its length a TBS of the
    1A table at N_PRB^1A 2: the mcs is found for it), on resource blocks from `first` (default the lowest); its DCI at
    aggregation L. cfi: default the smallest whose control region holds L CCEs"""
    if cfi is None:
        cfi = next(c for c in (1, 2, 3) if len(phy.control_regs(pci, n_rb, phy.control_symbols(c, n_rb), phy.phich_groups(phich_ng, n_rb),
                                                              ports)) >= 9*L)
    nfft = {v: k for k, v in phy.N_RB.items()}[n_rb]
    dci = None
    if sib is not None:
        tbs = tbs_table()
        if mcs is None:
            mcs = next(m for m, row in sorted(tbs.items()) if row[1] == len(sib))
        count = 4 if n_rb > 6 else 2 # (the allocation holds the block at the table's N_PRB^1A of 2)
        dci = dci_1a(n_rb, 0 if first is None else first, count, mcs)
    grids = []
    for f in range(frames):
        sfn = (sfn0 + f) % 1024
        for sf in range(10):
            on = sib is not None and sf == 5 and sfn % 2 == 0
            grids.append(subframe(pci, sf, sfn, n_rb, ports, cfi, phich_ng, dci if on else None, sib if on else None, L=L))
    return modulate(np.concatenate(grids, axis=1), nfft)

def channel(y, gains=None, cfo=0.0, fs=None, delay=0, snr_db=None, seed=0):
    """The ports' samples (ports, n) through a flat gain each, summed; then a carrier offset [Hz at fs], `delay` samples of
    noise before, and noise at snr_db (per sample, of the summed signal)"""
    rng = np.random.default_rng(seed)
    ports = len(y)
    if gains is None:
        gains = np.exp(2j*np.pi*rng.random(ports))*(0.6 + 0.4*rng.random(ports))
    x = np.sum(np.asarray(gains)[:, None]*y, axis=0)
    x = np.concatenate((np.zeros(delay, complex), x))
    if cfo:
        x = x*np.exp(2j*np.pi*cfo*np.arange(len(x))/fs)
    if snr_db is not None:
        p = np.mean(np.abs(x[delay:])**2)
        n = np.sqrt(p/10**(snr_db/10)/2)
        x = x + n*(rng.standard_normal(len(x)) + 1j*rng.standard_normal(len(x)))
    return x.astype(np.complex64)
