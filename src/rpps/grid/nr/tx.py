"""A 5G NR downlink to test the receive side with, and a LEO satellite pass to put it through

    y = rp.grid.nr.tx.downlink(pci=17, nfft=512, frames=8, sib=bits)       # 7.68 MS/s, from slot 0 of SFN 0
    x = rp.grid.nr.tx.leo(y, fs=7.68e6, fc=2.185e9, t0=-2.0)               # 2 s before closest approach

SS/PBCH blocks every 20 ms (even frames, first half frame), blocks `ssb` (case A); the MIB of each frame; in slot n0 of
each block's Type0-PDCCH occasion a DCI 1_0 to the SI-RNTI in CORESET 0 (aggregation `L`), and the transport block `sib`
on the PDSCH it allocates, in the occasion of every block sent (mapping from the default table A, rate matched around the SS/PBCH block). 15 kHz subcarriers,
one antenna port. `leo` moves the receiver under a satellite in a circular orbit: the delay follows the slant range (the
samples stretched and squeezed by it) and the carrier the Doppler, -fc dr/dt / c, tens of kHz at S-band, changing by
hundreds of Hz a second at closest approach (3GPP TR 38.811 6.9 gives the same orders).
"""
import numpy as np

from . import phy

C_LIGHT = 299792458.0
R_EARTH, MU = 6371e3, 3.986004418e14

def _bits(v, n):
    return [(int(v) >> (n - 1 - i)) & 1 for i in range(n)]

def dci_10_si(n_rb, first, count, tdra=0, mcs=0, rv=0, sii=0):
    w = int(np.ceil(np.log2(n_rb*(n_rb + 1)/2)))
    return np.array(_bits(phy.riv(first, count, n_rb), w) + _bits(tdra, 4) + [0] + _bits(mcs, 5) + _bits(rv, 2) + [sii] + [0]*15, np.uint8)

def sib_size(n_rb_alloc, tdra_row, dmrs_pos, mcs):
    """The TBS of an allocation (what a DCI with these fields carries)"""
    t = phy.tdra(tdra_row, dmrs_pos)
    nd = len(phy.dmrs_symbols(t["mapping"], t["s"], t["l"], dmrs_pos))
    qm, r = phy.mcs(mcs)
    return phy.tbs(12*t["l"] - 12*nd, n_rb_alloc, qm, r)

def downlink(pci=17, nfft=512, frames=4, sib=None, ssb=(0,), coreset0=0, ss0=0, k_ssb=0, dmrs_pos=2, mcs=2, first=0,
             count=None, tdra=0, L=4, cce=0, sfn0=0, fc=0.0):
    """Samples (from slot 0 of frame sfn0) of `frames` radio frames. sib: the transport block (its length: the TBS the
    allocation carries, `sib_size`). fc: the SS/PBCH block centre's absolute frequency, for the phase NR's upconversion
    gives each symbol (38.211 5.4; 0: none)"""
    from ... import code, link
    c0 = phy.coreset0(coreset0)
    n_rb, nsym = c0["rbs"], c0["symbols"]
    c0bin = phy.coreset0_first_bin(nfft, k_ssb, c0["offset"])
    count = n_rb if count is None else count
    slots = []
    pbch_code = code.load("block/polar/nr_pbch").encoder
    for f in range(frames):
        sfn = (sfn0 + f) % 1024
        for s in range(10):
            G = np.zeros((14, nfft), complex)
            ssb_here = []
            if sfn % 2 == 0: # Blocks every 20 ms, in the first half frame
                mib = phy.build_mib(0, k_ssb, dmrs_pos, coreset0, ss0)
                e = pbch_code.encode(phy.payload_encode(mib, sfn, 0, k_ssb, pci)).astype(np.uint8)
                for i in ssb:
                    slot, l0 = phy.ssb_place(pci, i)
                    if slot != s:
                        continue
                    ssb_here.append(l0)
                    b = phy.ssb_bins(nfft)
                    G[l0, b[56:183]] = phy.pss(pci % 3)
                    G[l0 + 2, b[56:183]] = phy.sss(pci//3, pci % 3)
                    pk, pl, dk, dl = phy.ssb_map(pci)
                    bits = e ^ phy.pbch_scramble(pci, i)
                    G[l0 + pl, b[pk]] = phy.qam(bits, 2)
                    G[l0 + dl, b[dk]] = phy.pbch_dmrs(pci, i)
            occ = [i for i in ssb if phy.type0_slots(ss0, i, nsym)[0][0] == s and sfn % 2 == phy.type0_slots(ss0, i, nsym)[1]]
            if sib is not None and occ: # SIB1 in slot n0 of every block's Type0 occasion
                fsym = phy.type0_slots(ss0, occ[0], nsym)[2]
                dci = dci_10_si(n_rb, first, count, tdra, mcs)
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
                (pb, ps), (qb, qs, qi) = phy.pdsch_res(nfft, c0bin, first, count, t["s"], t["l"], t["mapping"], dmrs_pos, ssb_here)
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
                G *= np.conj(phy.symbol_phase(fc, nfft))[:, None]
            slots.append(G)
    return phy.modulate(np.concatenate(slots), nfft)

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
