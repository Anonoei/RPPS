"""A 5G NR downlink to test the receive side with, and a LEO satellite pass to put it through

    y = rp.grid.nr.tx.downlink(pci=17, nfft=512, frames=8, sib=bits)       # 7.68 MS/s, from slot 0 of SFN 0
    y = rp.grid.nr.tx.downlink(scs=30e3, nfft=512, sib=bits)              # 30 kHz, 15.36 MS/s
    x = rp.grid.nr.tx.leo(y, fs=7.68e6, fc=2.185e9, t0=-2.0)               # 2 s before closest approach

SS/PBCH blocks every 20 ms (even frames, half frame `hrf`), blocks `ssb` of the pattern's candidates; the MIB of each
frame; in slot n0 of each block's Type0-PDCCH occasion a DCI 1_0 to the SI-RNTI in CORESET 0 (aggregation `L`), and the
transport block `sib` on the PDSCH it allocates, in the occasion of every block sent (mapping from the default table A,
rate matched around the SS/PBCH block). The block at `scs`, the control and shared channels at `scs_common`, one antenna
port. `leo` moves the receiver under a satellite in a circular orbit: the delay follows the slant range (the samples
stretched and squeezed by it) and the carrier the Doppler, -fc dr/dt / c, tens of kHz at S-band, changing by hundreds of
Hz a second at closest approach (3GPP TR 38.811 6.9 gives the same orders).
"""
import numpy as np

from . import phy

C_LIGHT = 299792458.0
R_EARTH, MU = 6371e3, 3.986004418e14

def _bits(v, n):
    return [(int(v) >> (n - 1 - i)) & 1 for i in range(n)]

def dci_10_si(n_rb, first, count, tdra=0, mcs=0, rv=0, sii=0, vrb=0):
    w = int(np.ceil(np.log2(n_rb*(n_rb + 1)/2)))
    return np.array(_bits(phy.riv(first, count, n_rb), w) + _bits(tdra, 4) + [vrb] + _bits(mcs, 5) + _bits(rv, 2) + [sii] + [0]*15, np.uint8)

def sib_size(n_rb_alloc, tdra_row, dmrs_pos, mcs):
    """The TBS of an allocation (what a DCI with these fields carries)"""
    t = phy.tdra(tdra_row, dmrs_pos)
    nd = len(phy.dmrs_symbols(t["mapping"], t["s"], t["l"], dmrs_pos))
    qm, r = phy.mcs(mcs)
    return phy.tbs(12*t["l"] - 12*nd, n_rb_alloc, qm, r)

def downlink(pci=17, nfft=512, frames=4, sib=None, ssb=(0,), coreset0=0, ss0=0, k_ssb=0, dmrs_pos=2, mcs=2, first=0,
             count=None, tdra=0, L=4, cce=0, sfn0=0, fc=0.0, scs=15e3, scs_common=None, pattern=None, lmax=None, hrf=0,
             vrb=0, min_bw=5):
    """Samples of a downlink, scs x nfft a second

    From slot 0 of frame sfn0, `frames` radio frames. sib: the transport block (its length: the TBS the allocation carries,
    `sib_size`). fc: the SS/PBCH block centre's absolute frequency, for the phase NR's upconversion gives each symbol (38.211
    5.4; 0: none). scs, scs_common (default: scs): the block's subcarrier spacing and the other channels'; pattern: "A", "B"
    or "C" (default A at 15 kHz, C at 30); lmax: candidate blocks in a half frame, 4 or 8 (default 4 at 15 kHz, 8 at 30);
    hrf: the half frame the blocks are in; vrb: interleaved virtual blocks"""
    from ... import code, link
    mu = phy.mu_of(scs)
    mu_c = mu if scs_common is None else phy.mu_of(scs_common)
    scs_common = 15e3*(1 << mu_c)
    size = nfft << mu # samples of a subframe / 15
    nfft_c = size >> mu_c
    pattern = ("C" if mu else "A") if pattern is None else pattern
    lmax = (8 if mu else 4) if lmax is None else lmax
    c0 = phy.coreset0(coreset0, scs, scs_common, min_bw)
    n_rb, nsym = c0["rbs"], c0["symbols"]
    c0bin = phy.coreset0_first_bin(nfft_c, k_ssb, c0["offset"], scs, scs_common)
    if c0bin is None or c0bin < 0 or c0bin + 12*n_rb > nfft_c:
        raise ValueError(f"CORESET 0 (k_SSB {k_ssb}) does not lie on the grid of {nfft_c} subcarriers")
    count = n_rb if count is None else count
    pbch_code = code.load("block/polar/nr_pbch").encoder
    per_c = 10 << mu_c # slots a frame, at the channels' spacing and at the block's
    per_b = 10 << mu
    Gc = np.zeros((frames*per_c, 14, nfft_c), complex)
    Gb = np.zeros((frames*per_b, 14, nfft), complex)
    held = {} # slot of Gc -> the footprints of the blocks in it
    for f in range(frames):
        sfn = (sfn0 + f) % 1024
        if sfn % 2: # Blocks every 20 ms
            continue
        mib = phy.build_mib(int(mu_c), k_ssb, dmrs_pos, coreset0, ss0)
        e = pbch_code.encode(phy.payload_encode(mib, sfn, hrf, k_ssb, pci)).astype(np.uint8)
        for i in ssb:
            slot, l0 = phy.ssb_place(pattern, i)
            slot += (5 << mu)*hrf
            G = Gb[f*per_b + slot]
            b = phy.ssb_bins(nfft)
            G[l0, b[56:183]] = phy.pss(pci % 3)
            G[l0 + 2, b[56:183]] = phy.sss(pci//3, pci % 3)
            pk, pl, dk, dl = phy.ssb_map(pci)
            v = i & (3 if lmax == 4 else 7)
            G[l0 + pl, b[pk]] = phy.qam(e ^ phy.pbch_scramble(pci, v), 2)
            G[l0 + dl, b[dk]] = phy.pbch_dmrs(pci, v + (4*hrf if lmax == 4 else 0))
            # What it covers of the channels' grid
            lead = phy.cp_lengths(nfft, mu)[1]
            at = slot*phy.slot_len(nfft) + int(phy.symbol_starts(nfft, mu)[l0])
            begin, end = at - lead, at + 3*(nfft + lead) + nfft
            cp_c = phy.cp_lengths(nfft_c, mu_c)[1]
            for s in range(per_c):
                st = s*phy.slot_len(nfft_c) + phy.symbol_starts(nfft_c, mu_c)
                hit = np.nonzero((st + nfft_c > begin + cp_c//2) & (st - cp_c < end - cp_c//2))[0]
                if len(hit):
                    half = 120*(1 << mu)//(1 << mu_c)
                    held.setdefault(f*per_c + s, []).append((int(hit[0]), int(hit[-1]) + 1, nfft_c//2 - half, nfft_c//2 + half))
    if sib is not None:
        done = set()
        for f in range(frames):
            sfn = (sfn0 + f) % 1024
            for i in ssb: # SIB1 in slot n0 of every block's Type0 occasion
                (n0, _), parity, fsym = phy.type0_slots(ss0, i, nsym, mu_c)
                at = f*per_c + n0
                if sfn % 2 != parity or at in done:
                    continue
                done.add(at)
                s, G = n0, Gc[at]
                dci = dci_10_si(n_rb, first, count, tdra, mcs, vrb=vrb)
                K = len(dci) + 24
                c = code.load("block/polar/nr_dci", K=K, E=108*L, mask=phy.SI_RNTI).encoder
                bits = c.encode(dci).astype(np.uint8) ^ phy.gold(pci, 108*L)
                (db, ds), (mb, ms, mi) = phy.pdcch_res(c0bin, n_rb, nsym, fsym, pci, cce, L)
                G[ds, db] = phy.qam(bits, 2)
                for sym in np.unique(ms):
                    seq = phy.pdcch_dmrs(pci, s, sym, 3*n_rb)
                    sel = ms == sym
                    G[sym, mb[sel]] = seq[mi[sel]]
                t = phy.tdra(tdra + 1, dmrs_pos)
                prbs = np.arange(first, first + count)
                if vrb:
                    prbs = phy.vrb_interleaver(n_rb)[prbs]
                (pb, ps), (qb, qs, qi) = phy.pdsch_res(c0bin, prbs, t["s"], t["l"], t["mapping"], dmrs_pos, held.get(at, ()))
                qm, rate = phy.mcs(mcs)
                tb = np.asarray(sib, np.uint8)
                crc, ldpc = phy.ldpc_for(len(tb), rate, qm*len(pb), 0, qm)
                cw = ldpc.encoder.encode(np.r_[tb, np.asarray(link.load(f"crc/{crc}").encoder.field(tb)).astype(np.uint8)])
                bits = cw.astype(np.uint8) ^ phy.gold((phy.SI_RNTI << 15) + pci, len(cw))
                G[ps, pb] = phy.qam(bits, qm)
                for sym in np.unique(qs):
                    seq = phy.pdsch_dmrs(pci, s, sym, int(qi.max()) + 1)
                    sel = qs == sym
                    G[sym, qb[sel]] = seq[qi[sel]]
    if fc:
        for G, n, m in ((Gb, nfft, mu), (Gc, nfft_c, mu_c)):
            for s in range(len(G)):
                G[s] *= np.conj(phy.symbol_phase(fc, n, m, s))[:, None]
    return (phy.modulate(Gb.reshape(-1, nfft), nfft, mu) + phy.modulate(Gc.reshape(-1, nfft_c), nfft_c, mu_c))

def awgn(x, snr_db, seed=0, ref=None):
    rng = np.random.default_rng(seed)
    p = np.mean(np.abs(x if ref is None else ref)**2)
    n = np.sqrt(p/10**(snr_db/10)/2)
    return (x + n*(rng.standard_normal(len(x)) + 1j*rng.standard_normal(len(x)))).astype(np.complex64)

def orbit(t, altitude=600e3):
    """Slant range [m] to a satellite in a circular orbit passing overhead, t seconds from closest approach"""
    a = R_EARTH + altitude
    w = np.sqrt(MU/a**3)
    return np.sqrt(R_EARTH**2 + a**2 - 2*R_EARTH*a*np.cos(w*np.asarray(t, float)))

def doppler(t, fc, altitude=600e3):
    """The carrier's Doppler shift [Hz] at t (from closest approach)"""
    dt = 1e-3
    return -fc*(orbit(t + dt, altitude) - orbit(t - dt, altitude))/(2*dt)/C_LIGHT

def leo(y, fs, fc, t0=0.0, altitude=600e3, delay=0, taps=24):
    """y's samples received from a LEO pass starting t0 s from closest approach: delayed by the slant range (less its value
    at the start) and shifted by the carrier's phase over it; `delay` samples of nothing before"""
    n = np.arange(len(y))
    t = t0 + n/fs
    tau = (orbit(t, altitude) - orbit(t0, altitude))/C_LIGHT
    src = n - tau*fs # Where in y each received sample comes from
    i0 = np.floor(src).astype(np.int64)
    frac = src - i0
    out = np.zeros(len(y), complex)
    k = np.arange(-taps//2 + 1, taps//2 + 1)
    yp = np.concatenate((np.zeros(taps), y, np.zeros(taps)))
    for kk in k: # Windowed sinc interpolation
        w = np.sinc(frac - kk)*(0.5 + 0.5*np.cos(np.pi*(frac - kk)/(taps/2)))
        out += w*yp[np.clip(i0 + kk + taps, 0, len(yp) - 1)]
    out *= np.exp(-2j*np.pi*fc*tau)
    return np.concatenate((np.zeros(delay, complex), out))
