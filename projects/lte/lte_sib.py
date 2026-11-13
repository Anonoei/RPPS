"""LTE control channel and system information: PCFICH -> PDCCH (SI-RNTI) -> PDSCH -> SIB1

    python projects/lte/lte_sib.py data/lte_b13.cf64 --fs 15.36 --cf 753 --carrier 751.0

Runs the cell search and the MIB first (lte_cell.py, lte_pbch.py). SIB1 is sent in subframe 5 of every even radio frame (36.331 5.2.1.2), scheduled by
a PDCCH whose CRC is XORed with the SI-RNTI 0xFFFF in the common search space. References (3GPP TS 36.211 v19.3.0 6.7-6.9, 36.212 5.3.3-5.3.4, 36.213 7.1):
  PCFICH (6.7): 4 resource-element groups (REGs) in symbol 0 at k = kbar + floor(i*N_RB/2)*6, kbar = 6*(PCI mod 2 N_RB); 32 bits = codeword of CFI 1/2/3
  PHICH (6.9): ceil(Ng*N_RB/8) groups of 3 REGs in symbol 0, REG number (PCI + m' + floor(i*n0/3)) mod n0 among the n0 REGs not used by PCFICH
  PDCCH (6.8): the remaining REGs of the first CFI symbols, in time-first order (for each frequency position: symbol 0..CFI-1); quadruplets of QPSK
               symbols, one per REG, 9 REGs per CCE; before mapping they are permuted by the 32-column sub-block interleaver and cyclically
               shifted by the PCI; bits scrambled with Gold c_init = floor(ns/2)*2^9 + PCI
  common search space: CCEs 0..15, aggregation 4 (candidates at 0,4,8,12) or 8 (at 0, 8); DCI = payload + CRC16 XOR RNTI, tail-biting 1/3 code
"""
import argparse
from collections import Counter

import numpy as np

import lte_cell
import lte_pbch
import lte_turbo
from lte_pbch import gold, crs_values, crs_shift, CRS_SYMBOLS, sc_bin, subblock_perm, rate_dematch, viterbi_tailbiting, crc16

CFI_WORDS = {
    1: [0, 1, 1]*10 + [0, 1],
    2: [1, 0, 1]*10 + [1, 0],
    3: [1, 1, 0]*10 + [1, 1],
}
SI_RNTI = 0xFFFF
P_RNTI = 0xFFFE

def symbol_len(l, nfft):
    scale = nfft/2048
    return int(round(160*scale)) + nfft if l == 0 else int(round(144*scale)) + nfft

def subframe_symbol_starts(S, nfft):
    """Useful-part start sample of the 14 OFDM symbols of the subframe whose first sample is S"""
    out = []
    for ns in (0, 1):
        for l in range(7):
            out.append(lte_pbch.symbol_start(S, ns, l, nfft))
    return out

def subframe_grid(y, S, nfft, n_rb, pci, sf):
    """FFT every symbol of subframe sf (first sample S) and estimate the channel of all four CRS ports on every resource element:
    least squares at the CRS, linear in frequency between them, a straight line in time over the symbols that carry that port (a static receiver)
    Returns G (14, 12*n_rb) and H (4, 14, 12*n_rb)."""
    st = subframe_symbol_starts(S, nfft)
    G = np.array([lte_pbch.grid_symbol(y, u, nfft, n_rb) for u in st])
    ks = np.arange(12*n_rb)
    vs = pci % 6
    H = np.zeros((4, 14, 12*n_rb), complex)
    for p in range(4):
        ts, hs = [], []
        for slot in (0, 1):
            ns = 2*sf + slot
            for l in CRS_SYMBOLS[p]:
                sym = slot*7 + l
                v = (crs_shift(p, ns, l) + vs) % 6
                kk = 6*np.arange(2*n_rb) + v
                h = G[sym][kk]/crs_values(pci, ns, l, n_rb)
                hi = np.interp(ks, kk, h.real) + 1j*np.interp(ks, kk, h.imag)
                ts.append(st[sym]); hs.append(hi)
        t = np.array(ts, float); Hs = np.array(hs)
        tm = t.mean()
        A = np.vstack((np.ones_like(t), t - tm)).T
        coef = np.linalg.lstsq(A, Hs, rcond=None)[0]
        for sym in range(14):
            H[p, sym] = coef[0] + coef[1]*(st[sym] - tm)
    return G, H

def diversity4(G, H, res):
    """Equalize a list of resource elements (k, sym) taken in quartets as SFBC-FSTD on 4 ports (36.211 6.3.4.3): ports (0,2) over the first pair, (1,3) over the second"""
    r = np.array([G[l][k] for k, l in res])
    h = lambda p: np.array([H[p][l][k] for k, l in res])
    h0, h1, h2, h3 = h(0), h(1), h(2), h(3)
    z = np.zeros(len(res), complex)
    for q in range(0, len(res), 4):
        a, b, c, d = r[q:q + 4]
        ha, hb = (h0[q] + h0[q + 1])/2, (h2[q] + h2[q + 1])/2
        hc, hd = (h1[q + 2] + h1[q + 3])/2, (h3[q + 2] + h3[q + 3])/2
        z[q] = np.conj(ha)*a + hb*np.conj(b)
        z[q + 1] = np.conj(ha)*b - hb*np.conj(a)
        z[q + 2] = np.conj(hc)*c + hd*np.conj(d)
        z[q + 3] = np.conj(hc)*d - hd*np.conj(c)
    return z

def reg_starts(l, ports=4):
    """Subcarrier of the first RE of each REG in symbol l: 6 subcarriers where CRS is present (symbol 0, and 1 for 4 ports), else 4 (6.2.4)"""
    return 6 if (l == 0 or (l == 1 and ports == 4)) else 4

def reg_res(k0, l, pci, ports=4):
    """The four non-CRS resource elements (k, l) of the REG starting at subcarrier k0"""
    if reg_starts(l, ports) == 6:
        vs = pci % 6
        return [(k, l) for k in range(k0, k0 + 6) if (k - vs) % 3 != 0]
    return [(k, l) for k in range(k0, k0 + 4)]

def control_region(pci, n_rb, cfi, n_groups, ports=4):
    """The PDCCH REGs of a subframe, in the order the standard fills them (time first), as a list of REG start (k0, l); also the PCFICH REGs and the PHICH REGs"""
    nsc = 12*n_rb
    kbar = 6*(pci % (2*n_rb))
    pcfich = [((kbar + (i*n_rb//2)*6) % nsc, 0) for i in range(4)]
    sym0 = [k for k in range(0, nsc, 6) if (k, 0) not in pcfich]
    n0 = len(sym0)
    phich = []
    for m in range(n_groups):
        for i in range(3):
            n = (pci + m + (i*n0)//3) % n0
            phich.append((sym0[n], 0))
    used = set(pcfich) | set(phich)
    reg = []
    for k in range(nsc):
        for l in range(cfi):
            step = reg_starts(l, ports)
            if k % step == 0 and (k, l) not in used:
                reg.append((k, l))
    return reg, pcfich, phich

def decode_pcfich(G, H, pci, n_rb, sf):
    """CFI from the PCFICH: equalize the 16 symbols, descramble, correlate with the three codewords. Returns (cfi, margin, soft)"""
    nsc = 12*n_rb
    kbar = 6*(pci % (2*n_rb))
    res = []
    for i in range(4):
        res += reg_res((kbar + (i*n_rb//2)*6) % nsc, 0, pci)
    z = diversity4(G, H, res)
    s = lte_pbch.soft_bits(z)
    ns = 2*sf
    c = gold(((ns//2 + 1)*(2*pci + 1) << 9) + pci, 32)
    s = s*(1 - 2.0*c)
    sc = {cfi: float(np.sum(s*(1 - 2.0*np.array(w)))) for cfi, w in CFI_WORDS.items()}
    order = sorted(sc, key=sc.get, reverse=True)
    return order[0], sc[order[0]]/max(abs(sc[order[1]]), 1e-9), sc

def pdcch_soft_cces(G, H, pci, n_rb, cfi, n_groups, sf):
    """Soft bits of every CCE of the control region: equalize all PDCCH REGs, undo the cyclic shift and the quadruplet interleaver, descramble.
    Returns an array (N_CCE, 72)."""
    reg, _, _ = control_region(pci, n_rb, cfi, n_groups)
    M = len(reg)
    res = []
    for k0, l in reg:
        res += reg_res(k0, l, pci)
    z = diversity4(G, H, res)
    wbar = lte_pbch.soft_bits(z).reshape(M, 8) # one quadruplet = 4 QPSK symbols = 8 bits per REG
    w = np.zeros_like(wbar)
    # the standard shifts w_bar(i) = w((i + PCI) mod M), so undo it: w(j) = w_bar((j - PCI) mod M)
    for j in range(M):
        w[j] = wbar[(j - pci) % M]
    perm = list(subblock_perm(M)) # output j came from input perm[j]
    zq = np.zeros_like(wbar)
    for j, src in enumerate(perm):
        zq[src] = w[j]
    bits = zq.reshape(-1)
    c = gold(((2*sf)//2 << 9) + pci, len(bits))
    bits = bits*(1 - 2.0*c)
    n_cce = M//9
    return bits[:n_cce*72].reshape(n_cce, 72), M

def try_dci(llr, K):
    """Decode E soft bits as a DCI of K payload bits: returns (payload bits, rnti) where rnti = parity bits XOR the CRC of the payload (0..65535)"""
    D = K + 16
    bits, _ = viterbi_tailbiting(rate_dematch(llr, D))
    calc = crc16(bits[:K])
    rnti = 0
    for a, b in zip(bits[K:], calc):
        rnti = (rnti << 1) | (int(a) ^ int(b))
    return bits[:K], rnti

def blind_pdcch(cces, rntis=(SI_RNTI, P_RNTI), sizes=range(10, 61)):
    """Try every common-search-space candidate (aggregation 4 at CCE 0,4,8,12; 8 at 0,8) with every payload size; keep those whose CRC reduces to a wanted RNTI"""
    hits = []
    n = len(cces)
    for L, starts in ((4, (0, 4, 8, 12)), (8, (0, 8))):
        for s in starts:
            if s + L > n:
                continue
            llr = cces[s:s + L].reshape(-1)
            for K in sizes:
                bits, rnti = try_dci(llr, K)
                if rnti in rntis:
                    hits.append(dict(L=L, cce=s, K=K, rnti=rnti, bits=bits))
    return hits

def bits_to_int(b):
    return int("".join(str(int(x)) for x in b), 2)

def parse_dci_1a(bits, n_rb):
    """DCI 1A (36.212 5.3.3.1.3) with an SI/P/RA-RNTI: flag(1) VRB flag(1) RIV(ceil log2(N(N+1)/2)) MCS(5) HARQ(3) NDI(1) RV(2) TPC(2)"""
    nriv = int(np.ceil(np.log2(n_rb*(n_rb + 1)/2)))
    p = 0
    def take(n):
        nonlocal p
        v = bits_to_int(bits[p:p + n]) if p + n <= len(bits) else None
        p += n
        return v
    d = dict(flag=take(1), distributed=take(1), riv=take(nriv), mcs=take(5), harq=take(3), ndi=take(1), rv=take(2), tpc=take(2))
    d["pad"] = len(bits) - p
    return d

def riv_decode(riv, n):
    """Start and length of a contiguous allocation from the resource indication value (36.213 7.1.6.3): riv = N(L-1) + S or N(N-L+1) + (N-1-S)"""
    L = riv//n + 1; S = riv % n
    if S + L > n:
        L = n - L + 2; S = n - 1 - S
    return S, L

# TBS table 36.213 7.1.7.2.1-1, columns N_PRB 1..10, rows I_TBS 0..26 (DCI 1A with an SI-RNTI uses I_TBS = I_MCS and N_PRB^1A = 2 or 3 by the TPC low bit)
TBS = {
    0: [16, 32, 56, 88, 120, 152, 176, 208, 224, 256],
    1: [24, 56, 88, 144, 176, 208, 224, 256, 328, 344],
    2: [32, 72, 144, 176, 208, 256, 296, 328, 376, 424],
    3: [40, 104, 176, 208, 256, 328, 392, 440, 504, 568],
    4: [56, 120, 208, 256, 328, 408, 488, 552, 632, 696],
    5: [72, 144, 224, 328, 424, 504, 600, 680, 776, 872],
    6: [88, 176, 256, 392, 504, 600, 712, 808, 936, 1032],
    7: [104, 224, 328, 472, 584, 712, 840, 968, 1096, 1224],
    8: [120, 256, 392, 536, 680, 808, 968, 1096, 1256, 1384],
    9: [136, 296, 456, 616, 776, 936, 1096, 1256, 1416, 1544],
    10: [144, 328, 504, 680, 872, 1032, 1224, 1384, 1544, 1736],
    11: [176, 376, 584, 776, 1000, 1192, 1384, 1608, 1800, 2024],
    12: [208, 440, 680, 904, 1128, 1352, 1608, 1800, 2024, 2280],
    13: [224, 488, 744, 1000, 1256, 1544, 1800, 2024, 2280, 2536],
    14: [256, 552, 840, 1128, 1416, 1736, 1992, 2280, 2600, 2856],
    15: [280, 600, 904, 1224, 1544, 1800, 2152, 2472, 2728, 3112],
    16: [328, 632, 968, 1288, 1608, 1928, 2280, 2600, 2984, 3240],
    17: [336, 696, 1064, 1416, 1800, 2152, 2536, 2856, 3240, 3624],
    18: [376, 776, 1160, 1544, 1992, 2344, 2792, 3112, 3624, 4008],
    19: [408, 840, 1288, 1736, 2152, 2600, 2984, 3496, 3880, 4264],
    20: [440, 904, 1384, 1864, 2344, 2792, 3240, 3752, 4136, 4584],
    21: [488, 1000, 1480, 1992, 2472, 2984, 3496, 4008, 4584, 4968],
    22: [520, 1064, 1608, 2152, 2664, 3240, 3752, 4264, 4776, 5352],
    23: [552, 1128, 1736, 2280, 2856, 3496, 4008, 4584, 5160, 5736],
    24: [584, 1192, 1800, 2408, 2984, 3624, 4264, 4968, 5544, 5992],
    25: [616, 1256, 1864, 2536, 3112, 3752, 4392, 5160, 5736, 6200],
    26: [712, 1480, 2216, 2984, 3752, 4392, 5160, 5992, 6712, 7480],
}
# (I_TBS 6, N_PRB 1 reads 328 in the spec document's table, a typo: 88 fits the column)

def pdsch_res(prbs, cfi, pci, sf, n_rb):
    """The resource elements (k, l) of a localized PDSCH in the order the standard fills them (36.211 6.3.5: increasing k over the assigned PRBs, then l),
    leaving out the PDCCH symbols, CRS of all four ports, and the 72 central subcarriers of the PSS/SSS symbols (slot 0, symbols 5 and 6, subframes 0 and 5)"""
    vs = pci % 6
    mid = (6*n_rb - 36, 6*n_rb + 36)
    res = []
    for l in range(cfi, 14):
        for prb in sorted(prbs):
            for k in range(12*prb, 12*prb + 12):
                if l in (0, 1, 4, 7, 8, 11) and (k - vs) % 3 == 0:
                    continue
                if sf in (0, 5) and l in (5, 6) and mid[0] <= k < mid[1]:
                    continue
                if sf == 0 and l in (7, 8, 9, 10) and mid[0] <= k < mid[1]:
                    continue
                res.append((k, l))
    return res

def pdsch_soft(G, H, res, pci, rnti, sf):
    """Equalize (4-port transmit diversity) and descramble (c_init = RNTI 2^14 + floor(ns/2) 2^9 + PCI): soft bits in transmission order"""
    z = diversity4(G, H, res)
    s = lte_pbch.soft_bits(z)
    c = gold((rnti << 14) + (((2*sf)//2) << 9) + pci, len(s))
    return s*(1 - 2.0*c), z

def sib1_dci_params(dci, n_rb):
    """From a DCI 1A with SI-RNTI: (first PRB, number of PRBs, transport block size) assuming localized allocation"""
    S, L = riv_decode(dci["riv"], n_rb)
    n1a = 2 if (dci["tpc"] & 1) == 0 else 3
    return S, L, TBS[dci["mcs"]][n1a - 1]

def decode_sib(G, H, pci, n_rb, cfi, sf, dci, d_acc=None):
    """One PDSCH transmission of SIB1: returns (payload bits or None, crc ok, QPSK-ness, accumulated streams, K)"""
    S, L, tbs = sib1_dci_params(dci, n_rb)
    res = pdsch_res(range(S, S + L), cfi, pci, sf, n_rb)
    s, z = pdsch_soft(G, H, res, pci, SI_RNTI, sf)
    K = tbs + 24
    d = lte_turbo.dematch(s, K, dci["rv"], d_acc)
    bits, _ = lte_turbo.turbo_decode(d, K)
    ok = lte_turbo.crc24a(bits) == [0]*24
    return (bits[:tbs] if ok else None), ok, lte_pbch.qpskness(z), d, K

def receive_subframe(y, S, nfft, n_rb, pci, sf, n_groups, rntis=(SI_RNTI, P_RNTI), want=SI_RNTI):
    """Everything for one subframe whose first sample is S: CFI, the DCIs for the wanted RNTIs and, for a format-1A DCI of `want`, the decoded PDSCH.
    Returns a dict (cfi, margin, hits, pdsch)."""
    G, H = subframe_grid(y, S, nfft, n_rb, pci, sf)
    cfi, margin, _ = decode_pcfich(G, H, pci, n_rb, sf)
    cces, M = pdcch_soft_cces(G, H, pci, n_rb, cfi, n_groups, sf)
    hits = blind_pdcch(cces, rntis)
    pdsch = []
    seen = set()
    for h in hits:
        key = (h["rnti"], tuple(int(b) for b in h["bits"]))
        if key in seen: # the aggregation-8 candidate also holds the aggregation-4 copy: same DCI, don't decode twice
            continue
        seen.add(key)
        if h["rnti"] == want and h["K"] >= 26:
            dci = parse_dci_1a(h["bits"], n_rb)
            if dci["flag"] != 1 or dci["distributed"] or dci["mcs"] not in TBS:
                pdsch.append(dict(dci=dci, ok=False, note="not a localized 1A"))
                continue
            bits, ok, qk, d, K = decode_sib(G, H, pci, n_rb, cfi, sf, dci)
            S_, L_, tbs_ = sib1_dci_params(dci, n_rb)
            pdsch.append(dict(dci=dci, ok=ok, bits=bits, qk=qk, S=S_, L=L_, tbs=tbs_))
    return dict(cfi=cfi, margin=margin, M=M, n_cce=len(cces), hits=hits, pdsch=pdsch)

def frame_positions(cell, sfn_of, pss_off):
    """{SFN: {0: first sample of subframe 0, 5: first sample of subframe 5}} from the PSS occurrences (each half-frame has one)"""
    pos = {}
    for k, s in sfn_of.items():
        pos.setdefault(s, {})[0] = int(cell["pss_pos"][k]) - pss_off
        if k + 1 < len(cell["pss_pos"]):
            pos[s][5] = int(cell["pss_pos"][k + 1]) - pss_off
    return pos

PERIOD = {"rf8": 8, "rf16": 16, "rf32": 32, "rf64": 64, "rf128": 128, "rf256": 256, "rf512": 512}
WINDOW = {"ms1": 1, "ms2": 2, "ms5": 5, "ms10": 10, "ms15": 15, "ms20": 20, "ms40": 40}

def si_windows(sib1):
    """The system-information windows (36.331 5.2.3): message n (0-based in schedulingInfoList) with window w starts at x = n w; the frames are
    SFN mod T = floor(x/10), the first subframe x mod 10. Returns a list of (n, T, w, frame residue, first subframe, sibs)."""
    w = WINDOW[sib1["si-WindowLength"]]
    out = []
    for n, si in enumerate(sib1["schedulingInfoList"]):
        x = n*w
        out.append((n, PERIOD[si["si-Periodicity"]], w, x//10, x % 10, si["sib-MappingInfo"]))
    return out

def in_window(sfn, sf, win):
    n, T, w, res, a, _ = win
    t = sfn*10 + sf
    for base in range((sfn//T - 1)*T, (sfn//T + 2)*T, T):
        t0 = (base + res)*10 + a
        if t0 <= t < t0 + w:
            return True
    return False

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path")
    ap.add_argument("--fs", type=float, required=True); ap.add_argument("--cf", type=float, required=True); ap.add_argument("--carrier", type=float, required=True)
    ap.add_argument("--asn", help="ASN.1 module built by lte_rrc.py (needs asn1tools): decode the messages and read the SI schedule from SIB1")
    ap.add_argument("--no-si", action="store_true", help="SIB1 only")
    a = ap.parse_args(argv)
    pb = lte_pbch.main([a.path, "--fs", str(a.fs), "--cf", str(a.cf), "--carrier", str(a.carrier)])
    cell = pb["cell"]; y, nfft, pci = cell["y"], cell["nfft"], cell["pci"]
    n_rb = {128: 6, 256: 15, 512: 25, 1024: 50, 1536: 75, 2048: 100}[nfft]
    scale = nfft/2048
    cp = int(round(144*scale)); cp0 = int(round(160*scale))
    pss_off = (cp0 + nfft) + 5*(cp + nfft) + cp
    slot_len = lte_pbch.symbol_start(0, 1, 0, nfft) - lte_pbch.symbol_start(0, 0, 0, nfft)
    frames = [(k, cell["pss_pos"][k]) for k, hyp, *_ in cell["found"] if hyp[0] == cell["n1"] and hyp[1] == 0]
    if pb.get("best") != 4 or not pb["results"]:
        print(f"\nPBCH decoded as {pb.get('best')} port(s) (or not at all): the control-channel and PDSCH receivers here only handle 4-port transmit diversity")
        return dict(pb=pb, out=[], n_rb=n_rb, n_groups=0, sib1=[])
    good = sorted((i, q, bits) for i, pp, q, bits, p in pb["results"] if pp == 4)
    sfn_of = {frames[i][0]: lte_pbch.parse_mib(bits)["sfn_msb"]*4 + q for i, q, bits in good}
    mib = lte_pbch.parse_mib(good[0][2])
    ng = {"1/6": 1/6, "1/2": 0.5, "1": 1, "2": 2}[mib["phich_resource"]]
    n_groups = int(np.ceil(ng*n_rb/8))
    pos = frame_positions(cell, sfn_of, pss_off)
    def start(sfn, sf):
        base = pos.get(sfn, {}).get(0 if sf < 5 else 5)
        if base is None:
            return None
        S = base + (sf % 5)*2*slot_len
        return S if S - 20 >= 0 and S + 2*slot_len + 100 <= len(y) else None
    print(f"\nSIB1: PHICH Ng = {ng}, {n_groups} groups (3 REGs each), normal duration")
    out = []
    payloads = {}
    for s_ in sorted(pos):
        if s_ % 2:
            continue
        S = start(s_, 5)
        if S is None:
            continue
        r = receive_subframe(y, S, nfft, n_rb, pci, 5, n_groups)
        out.append((s_, r))
        d = r["pdsch"][0] if r["pdsch"] else None
        print(f"  SFN {s_:4d}: CFI {r['cfi']} (margin {r['margin']:.1f}x), {r['M']} REGs = {r['n_cce']} CCEs; "
              + ((f"DCI mcs {d['dci']['mcs']} rv {d['dci']['rv']} PRB {d['S']}+{d['L']} TBS {d['tbs']}: QPSK-ness {d['qk']:.2f}, CRC24 {'OK' if d['ok'] else 'fail'}"
                  if 'tbs' in d else d['note']) if d else "no DCI"))
        if d and d["ok"]:
            payloads.setdefault(bytes(np.packbits(d["bits"])), []).append(s_)
    ok = sum(1 for _, r in out if r["pdsch"] and r["pdsch"][0]["ok"])
    print(f"\nCFI votes {dict(Counter(r['cfi'] for _, r in out))}; DCIs found in {sum(1 for _, r in out if r['hits'])}/{len(out)} SIB1 subframes; PDSCH CRC ok in {ok}")
    for pl, sfns in payloads.items():
        print(f"  SIB1 payload (in {len(sfns)} subframes): {pl.hex()}")
    res = dict(pb=pb, out=out, n_rb=n_rb, n_groups=n_groups, sib1=list(payloads))
    if a.asn and payloads:
        import lte_rrc
        codec = lte_rrc.load(a.asn)
        import pprint
        msg = lte_rrc.decode_bcch_dl_sch(codec, max(payloads, key=lambda k: len(payloads[k])))
        sib1 = msg["message"][1][1]
        print("\nSIB1:"); pprint.pprint(sib1, width=140)
        res["sib1_msg"] = sib1
        if not a.no_si:
            wins = si_windows(sib1)
            print("\nSI windows:", [(n + 1, f"every {T} frames", f"{w} ms", sibs) for n, T, w, r_, a_, sibs in wins])
            si = {}
            for s_ in sorted(pos):
                for sf in range(10):
                    if (s_ % 2 == 0 and sf == 5):
                        continue
                    wi = [w_ for w_ in wins if in_window(s_, sf, w_)]
                    S = start(s_, sf)
                    if not wi or S is None:
                        continue
                    r = receive_subframe(y, S, nfft, n_rb, pci, sf, n_groups, rntis=(SI_RNTI,))
                    for d in r["pdsch"]:
                        tag = f"msg {wi[0][0] + 1}"
                        if 'tbs' in d:
                            print(f"  SFN {s_} sf {sf}: {tag}: CFI {r['cfi']} mcs {d['dci']['mcs']} rv {d['dci']['rv']} PRB {d['S']}+{d['L']} TBS {d['tbs']} CRC24 {'OK' if d['ok'] else 'fail'}")
                        if d.get("ok"):
                            si.setdefault((wi[0][0], bytes(np.packbits(d["bits"]))), []).append((s_, sf))
                    if r["hits"] and not r["pdsch"]:
                        print(f"  SFN {s_} sf {sf}: {wi[0][0] + 1}: DCI sizes {[h['K'] for h in r['hits']]} (not a 1A we decode)")
            res["si"] = si
            for (n, pl), where in si.items():
                try:
                    m = lte_rrc.decode_bcch_dl_sch(codec, pl)
                    print(f"\nSI message {n + 1} (decoded in {len(where)} subframes, first SFN {where[0][0]} sf {where[0][1]}):"); pprint.pprint(m, width=140)
                except Exception as e:
                    print(f"\nSI message {n + 1}: ASN.1 decode failed ({str(e)[:80]}); bytes {pl.hex()}")
    return res

if __name__ == "__main__":
    main()
