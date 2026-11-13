"""LTE broadcast channel: PBCH -> MIB (channel bandwidth, PHICH config, system frame number, antenna-port count)

    python projects/lte/lte_pbch.py data/lte_b13.cf64 --fs 15.36 --cf 753 --carrier 751.0
    python projects/lte/lte_pbch.py --selftest

Needs the cell search of lte_cell.py first (it supplies the carrier offset, PCI and frame timing). References: 3GPP TS 36.211 (physical channels,
v19.x "j30": 6.6 PBCH, 6.10.1 cell reference signals, 7.2 Gold sequence) and TS 36.212 (5.3.1 BCH, 5.1.1 CRC, 5.1.3.1 tail-biting convolutional
code, 5.1.4.2 rate matching). Chain, in the order the receiver undoes it:
  1. subframe 0 starts 7 symbols before the PSS (the half-frame whose SSS said 0); slot 1 symbols 0-3 carry the PBCH on the central 72 subcarriers
  2. channel estimate from the cell-specific reference signals (CRS, a QPSK Gold sequence seeded by the PCI) of ports 0-3, interpolated
  3. equalize: 1 port directly; 2 ports Alamouti (SFBC) on pairs of resource elements; 4 ports SFBC-FSTD on quartets
  4. QPSK -> soft bits, descramble (Gold, c_init = PCI, one 480-bit quarter per radio frame: which quarter is the frame number mod 4)
  5. rate dematching: 1920 bits are the 120 coded bits repeated 16 times (each frame carries 4 repeats: one frame is decodable alone)
  6. tail-biting rate-1/3 K=7 Viterbi (generators 133 171 165 octal), 40 bits out = 24 MIB bits + 16-bit CRC, CRC masked by the port count
Statistic per assumption: the "QPSK-ness" -Re(sum z^4)/sum |z|^4 of the equalized symbols is ~1 when channel estimate and precoding are right and ~0
when wrong, before any decoding; the final judge is the CRC (1 in 65536 by chance, per mask).
"""
import argparse
import sys
from collections import Counter

import numpy as np

import rpps as rp

import lte_cell

# ---------------------------------------------------------------------------------------------------------------- sequences (36.211 7.2, 6.10.1)
def gold(c_init, n):
    """Gold sequence c(0..n-1) (36.211 7.2), from the rpps definition sequence/recurrence/lte_gold"""
    return rp.sequence.load("lte_gold", c_init=int(c_init)).bits(n)

def crs_values(n_id, ns, l, n_rb, n_cp=1):
    """Reference symbols r_{l,ns}(m'), m = 0..2*n_rb-1, for the cell n_id in slot ns (0..19), OFDM symbol l (36.211 6.10.1.1)"""
    c_init = (1 << 10)*(7*(ns + 1) + l + 1)*(2*n_id + 1) + 2*n_id + n_cp
    c = gold(c_init, 4*110)
    r = ((1 - 2.0*c[0::2]) + 1j*(1 - 2.0*c[1::2]))/np.sqrt(2)
    return r[np.arange(2*n_rb) + 110 - n_rb]

def crs_shift(port, ns, l):
    """v of 36.211 6.10.1.2: the comb offset of port 0-3 in symbol l of slot ns (before adding v_shift = PCI mod 6)"""
    if port == 0: return 0 if l == 0 else 3
    if port == 1: return 3 if l == 0 else 0
    if port == 2: return 3*(ns % 2)
    return 3 + 3*(ns % 2)

CRS_SYMBOLS = {0: (0, 4), 1: (0, 4), 2: (1,), 3: (1,)} # OFDM symbols of a slot that carry each port's CRS (normal cyclic prefix)

def sc_bin(k, nfft, n_rb):
    """FFT bin of resource-grid subcarrier k = 0..12*n_rb-1 (the DC bin is skipped)"""
    k = np.asarray(k)
    f = np.where(k < 6*n_rb, k - 6*n_rb, k - 6*n_rb + 1)
    return f % nfft

# ---------------------------------------------------------------------------------------------------------------- channel coding (36.212 5.1, 5.3.1)
CONV_GENS = (0o133, 0o171, 0o165)

_CRC16 = rp.link.load("crc", "crc16_xmodem")._enc # 36.212 5.1.1 gCRC16: D^16 + D^12 + D^5 + 1, zero start

def crc16(bits):
    """CRC (36.212 5.1.1) of a bit list, fed MSB first; returns 16 bits"""
    return [int(b) for b in _CRC16.to_bits(_CRC16.compute_bits(bits))]

CRC_MASKS = {1: [0]*16, 2: [1]*16, 4: [0, 1]*8} # table 5.3.1.1-1: number of antenna ports -> mask

_CONV = rp.code.Conv(CONV_GENS, 7, termination="tailbiting") # 36.212 5.1.3.1

def conv_encode(c):
    """Tail-biting rate-1/3 encoder (5.1.3.1): returns the three (K,) output streams"""
    return _CONV._enc.encode(np.asarray(c, np.uint8)).reshape(-1, 3).T.astype(np.uint8)

def subblock_perm(D):
    """For each output position of the sub-block interleaver (5.1.4.2.1) the input index, the <NULL> dummies left out (rpps interleave/block/lte_subblock)"""
    return rp.code.load("interleave", "lte_subblock", length=D)._enc.src

def _rm(D, E):
    return rp.code.load("block", "ratematch/lte_conv", D=D, E=E)._enc # 5.1.4.2


def rate_match(d, E):
    """(3, D) encoder streams -> E bits"""
    return _rm(d.shape[1], E).encode(np.asarray(d).T.reshape(-1)).astype(np.uint8)


def rate_dematch(llr, D=40):
    """Sum soft bits of repeated transmissions back onto the three streams: returns (3, D) LLRs"""
    return _rm(D, len(llr)).decode(np.asarray(llr, float)).reshape(D, 3).T

def viterbi_tailbiting(llr3):
    """Soft decode of the tail-biting code (llr3: (3, K), positive = bit 0) with the rpps Viterbi (three laps round the circle, the middle kept). Returns (bits, None)"""
    return _CONV._dec.decode(np.asarray(llr3, float).T.reshape(-1)).astype(np.uint8), None

def decode_bch(llr):
    """1920 descrambled soft bits -> (MIB bits, ports or None, crc bits). The port count is read from which mask makes the CRC zero."""
    bits, _ = viterbi_tailbiting(rate_dematch(llr, 40))
    got = list(bits[24:]); calc = crc16(bits[:24])
    xor = [a ^ b for a, b in zip(got, calc)]
    for ports, m in CRC_MASKS.items():
        if xor == m:
            return bits[:24], ports
    return bits[:24], None

# ---------------------------------------------------------------------------------------------------------------- MIB (36.331 6.2.2)
BANDWIDTH = {0: 6, 1: 15, 2: 25, 3: 50, 4: 75, 5: 100}
def parse_mib(b):
    """24 bits, MasterInformationBlock of 36.331 (v19 "j30", 6.2.2): dl-Bandwidth (3), phich-Config (duration 1 + resource 2), systemFrameNumber's 8
    MSBs (8), schedulingInfoSIB1-BR-r13 (5: 0 = no SIB1 for LTE-M/eMTC devices), systemInfoUnchanged-BR-r15 (1), partEARFCN-r17 (a 1-bit choice, then
    2 bits), spare (1)"""
    v = lambda a, n: int("".join(str(int(x)) for x in b[a:a + n]), 2)
    return dict(n_rb=BANDWIDTH.get(v(0, 3)), phich_duration=("normal", "extended")[v(3, 1)], phich_resource=("1/6", "1/2", "1", "2")[v(4, 2)],
                sfn_msb=v(6, 8), sib1_br=v(14, 5), si_unchanged_br=v(19, 1), part_earfcn=v(20, 3), spare=v(23, 1))

# ---------------------------------------------------------------------------------------------------------------- receiver
def symbol_start(slot_start, ns, l, nfft):
    """Sample of the useful part (after the prefix) of OFDM symbol l in slot ns (0/1) of a subframe whose first sample is slot_start"""
    scale = nfft/2048
    cp0 = int(round(160*scale)); cp = int(round(144*scale))
    slot_len = cp0 + nfft + 6*(cp + nfft)
    off = cp0 if l == 0 else (cp0 + nfft) + (l - 1)*(cp + nfft) + cp
    return slot_start + ns*slot_len + off

def grid_symbol(y, u, nfft, n_rb, early=8):
    """FFT of one OFDM symbol, with the window taken `early` samples before the useful start (inside the prefix, clear of the previous symbol),
    the resulting phase ramp removed. Returns the 12*n_rb subcarriers in order."""
    w = np.fft.fft(y[u - early:u - early + nfft])
    k = np.arange(12*n_rb)
    b = sc_bin(k, nfft, n_rb)
    f = np.where(k < 6*n_rb, k - 6*n_rb, k - 6*n_rb + 1)
    return w[b]*np.exp(2j*np.pi*f*early/nfft)

def channel_estimates(y, S0, nfft, n_rb, pci, k_lo, k_hi):
    """H[port][l] for the PBCH symbols l = 0..3 of slot 1 over subcarriers k_lo..k_hi-1: least-squares at each CRS, linear in frequency between
    them, then a straight line in time through the symbols of the same port in slots 0 and 1 (a stationary receiver: the channel barely moves in 0.5 ms)."""
    vs = pci % 6
    ks = np.arange(k_lo, k_hi)
    est = {}
    for p in range(4):
        pts = []
        for ns in (0, 1):
            for l in CRS_SYMBOLS[p]:
                u = symbol_start(S0, ns, l, nfft)
                G = grid_symbol(y, u, nfft, n_rb)
                v = (crs_shift(p, ns, l) + vs) % 6
                m = np.arange(2*n_rb)
                kk = 6*m + v
                sel = (kk >= k_lo - 12) & (kk < k_hi + 12)
                h = G[kk[sel]]/crs_values(pci, ns, l, n_rb)[sel]
                pts.append((u, np.interp(ks, kk[sel], h.real) + 1j*np.interp(ks, kk[sel], h.imag)))
        t = np.array([q[0] for q in pts], float); Hs = np.array([q[1] for q in pts])
        for l in range(4):
            u = symbol_start(S0, 1, l, nfft)
            if len(t) == 2:
                w = (u - t[0])/(t[1] - t[0]); h = Hs[0]*(1 - w) + Hs[1]*w
            else:
                A = np.vstack((np.ones_like(t), t - u)).T
                coef = np.linalg.lstsq(A, Hs, rcond=None)[0]
                h = coef[0]
            est[(p, l)] = h
    return est

def pbch_resource_elements(pci, n_rb):
    """The 240 (k, l) of the PBCH in order (36.211 6.6.4): increasing k, then l, in slot 1 symbols 0-3, skipping the REs where CRS of ports 0-3 may be"""
    vs = pci % 6
    out = []
    k0 = 6*n_rb - 36
    for l in range(4):
        for k in range(k0, k0 + 72):
            if l < 2 and (k - vs) % 3 == 0:
                continue
            out.append((k, l))
    return out

def equalize(y, S0, nfft, n_rb, pci, ports):
    """Equalized PBCH symbols (240 QPSK) for one subframe 0, under the hypothesis of 1, 2 or 4 transmit ports"""
    k0 = 6*n_rb - 36
    H = channel_estimates(y, S0, nfft, n_rb, pci, k0, k0 + 72)
    G = {l: grid_symbol(y, symbol_start(S0, 1, l, nfft), nfft, n_rb)[k0:k0 + 72] for l in range(4)}
    res = pbch_resource_elements(pci, n_rb)
    r = np.array([G[l][k - k0] for k, l in res])
    h = lambda p: np.array([H[(p, l)][k - k0] for k, l in res])
    h0, h1, h2, h3 = (h(p) for p in range(4))
    z = np.zeros(len(res), complex)
    if ports == 1:
        z = np.conj(h0)*r
    elif ports == 2:
        a, b = r[0::2], r[1::2]
        ha = (h0[0::2] + h0[1::2])/2; hb = (h1[0::2] + h1[1::2])/2 # a pair shares one channel (adjacent subcarriers)
        z[0::2] = np.conj(ha)*a + hb*np.conj(b)
        z[1::2] = np.conj(ha)*b - hb*np.conj(a)
    else:
        for q in range(0, len(res), 4):
            a, b, c, d = r[q:q + 4]
            ha, hb = (h0[q] + h0[q + 1])/2, (h2[q] + h2[q + 1])/2
            hc, hd = (h1[q + 2] + h1[q + 3])/2, (h3[q + 2] + h3[q + 3])/2
            z[q] = np.conj(ha)*a + hb*np.conj(b)
            z[q + 1] = np.conj(ha)*b - hb*np.conj(a)
            z[q + 2] = np.conj(hc)*c + hd*np.conj(d)
            z[q + 3] = np.conj(hc)*d - hd*np.conj(c)
    return z

def qpskness(z):
    """-Re(sum z^4)/sum |z|^4: 1 for points on the diagonals (QPSK as the standard maps it), ~0 for noise or a wrong equalizer"""
    return float(-np.sum(z**4).real/max(np.sum(np.abs(z)**4), 1e-30))

def soft_bits(z):
    """QPSK (36.211 7.1.2: b(2i) -> real, b(2i+1) -> imag, bit 0 = positive) to LLR-like values"""
    s = np.empty(2*len(z)); s[0::2] = z.real; s[1::2] = z.imag
    return s

def descramble(s, pci, quarter):
    c = gold(pci, 1920)
    return s*(1 - 2.0*c[quarter*480:(quarter + 1)*480])

def rate_dematch_frame(s):
    return s # one frame is 480 bits of the 1920; handled by position in decode_frames

# ---------------------------------------------------------------------------------------------------------------- selftest
def selftest():
    """Encode a random MIB the way a transmitter does, add noise, and take it back out. Checks the encoder, interleaver, scrambler and the decoder."""
    rng = np.random.default_rng(1)
    ok = 0; n = 40
    for trial in range(n):
        mib = rng.integers(0, 2, 24).astype(np.uint8)
        ports = (1, 2, 4)[trial % 3]
        crc = [a ^ b for a, b in zip(crc16(mib), CRC_MASKS[ports])]
        c = np.concatenate((mib, np.array(crc, np.uint8)))
        e = rate_match(conv_encode(c), 1920)
        quarter = trial % 4
        sc = gold(77, 1920)
        tx = e ^ sc
        rx_all = (1 - 2.0*tx) + rng.normal(0, 1.2, 1920) # ~ -1.6 dB per-bit SNR: decodable only because of the repetition and the code
        # the receiver sees one frame (a quarter): the other bits are erased (0)
        llr = np.zeros(1920); sl = slice(quarter*480, (quarter + 1)*480)
        llr[sl] = rx_all[sl]
        llr = llr*(1 - 2.0*sc)
        bits, p = decode_bch(llr)
        ok += (p == ports and np.array_equal(bits, mib))
    print(f"selftest: {ok}/{n} encode -> noisy single frame -> decode (ports and MIB recovered)")
    return ok == n

# ---------------------------------------------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", nargs="?")
    ap.add_argument("--fs", type=float); ap.add_argument("--cf", type=float); ap.add_argument("--carrier", type=float)
    ap.add_argument("--n-rb", type=int, default=0, help="Resource blocks (default: from the sample rate: 15.36 MS/s = 50)")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    cell = lte_cell.main([a.path, "--fs", str(a.fs), "--cf", str(a.cf), "--carrier", str(a.carrier)])
    y, nfft, pci = cell["y"], cell["nfft"], cell["pci"]
    n_rb = a.n_rb or {128: 6, 256: 15, 512: 25, 1024: 50, 1536: 75, 2048: 100}[nfft]
    scale = nfft/2048
    cp = int(round(144*scale)); cp0 = int(round(160*scale))
    pss_off = (cp0 + nfft) + 5*(cp + nfft) + cp
    frames = [(k, cell["pss_pos"][k]) for k, hyp, *_ in cell["found"] if hyp[0] == cell["n1"] and hyp[1] == 0]
    print(f"\nPBCH: PCI {pci}, {n_rb} RB, {len(frames)} subframe-0 occurrences")
    # --- statistic 1, per port hypothesis: QPSK-ness of the equalized symbols
    print("\nQPSK-ness of the equalized PBCH (1 = clean QPSK, ~0 = wrong), median over frames:")
    z_all = {}
    for ports in (1, 2, 4):
        zs = []
        for k, pos in frames:
            S0 = int(pos) - pss_off
            if S0 - 20 < 0 or symbol_start(S0, 1, 3, nfft) + nfft > len(y):
                continue
            zs.append(equalize(y, S0, nfft, n_rb, pci, ports))
        z_all[ports] = zs
        q = [qpskness(z) for z in zs]
        print(f"  {ports} port(s): {np.median(q):+.2f}  (min {min(q):+.2f}, max {max(q):+.2f}, {len(q)} frames)")
    # --- statistic 2: the CRC, per frame, for every (ports, quarter)
    print("\nCRC check per frame (ports hypothesis, scrambling quarter):")
    results = []
    for ports in (1, 2, 4):
        for i, z in enumerate(z_all[ports]):
            s = soft_bits(z)
            for q in range(4):
                llr = np.zeros(1920); llr[q*480:(q + 1)*480] = descramble(s, pci, q)
                bits, p = decode_bch(llr)
                if p is not None:
                    results.append((i, ports, q, bits, p))
    for ports in (1, 2, 4):
        n = len({i for i, pp, q, bits, p in results if pp == ports})
        print(f"  demodulated as {ports} port(s): {n}/{len(frames)} frames pass a CRC (by chance ~{len(frames)*4*3/65536:.2f})")
    best = max((1, 2, 4), key=lambda p: np.median([qpskness(z) for z in z_all[p]]))
    good = sorted((i, q, bits, p) for i, pp, q, bits, p in results if pp == best)
    if not good:
        print("  nothing decoded")
        return dict(results=results, z=z_all, cell=cell, best=best)
    print(f"\nBest demodulation: {best} port(s); the CRC mask says {Counter(p for *_, p in good).most_common(1)[0][0]} port(s)")
    sfn = [(i, parse_mib(b)["sfn_msb"]*4 + q) for i, q, b, p in good]
    diffs = Counter((s2 - s1) % 1024 - (i2 - i1) for (i1, s1), (i2, s2) in zip(sfn, sfn[1:]))
    print(f"System frame number 10-bit: {sfn[0][1]} .. {sfn[-1][1]} over {sfn[-1][0] - sfn[0][0] + 1} frames (10 ms each): "
          f"SFN steps minus frame steps {dict(diffs)} (all 0 = consistent)")
    mibs = Counter(tuple(parse_mib(b)[k] for k in ("n_rb", "phich_duration", "phich_resource", "sib1_br", "si_unchanged_br", "part_earfcn", "spare")) for i, q, b, p in good)
    print(f"MIB fields (bandwidth RB, PHICH duration, PHICH resource, SIB1-BR scheduling, SI-unchanged-BR, partEARFCN, spare): {dict(mibs)}")
    ev = [float(np.mean(np.abs(np.abs(z.real) - np.abs(z.imag))/np.mean(np.abs(z)))) for z in z_all[best]]
    print(f"Equalized-symbol spread across the diagonals: {np.mean(ev):.3f} (0 = perfect)")
    return dict(results=results, z=z_all, cell=cell, best=best)

if __name__ == "__main__":
    main()
