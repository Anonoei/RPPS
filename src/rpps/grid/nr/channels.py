"""NR downlink channels read off NrGrid's frames: PBCH (MIB), PDCCH (DCI 1_0 in CORESET 0), PDSCH (SIB1)

Each estimates its channel from its own DM-RS (least squares, linear between them in frequency, and in time for the
PDSCH) and publishes what it decoded from the first item of its slot (rpps.base.control.publish), as rpps.grid.lte's do:

    PBCH   slots NrGrid marks as holding the block: the MIB and the payload's timing (SFN, half frame, k_SSB's MSB);
           publishes sfn (and the output frame it was read in, frame_ref), coreset0, ss0, k_ssb, dmrs_pos, barred
    PDCCH  once the MIB is known: the Type0 occasions (38.213 13: slots n0, n0 + 1 of frames of the right parity) of
           CORESET 0, every candidate of aggregation 4, 8 and 16 (polar, block/polar/nr_dci), for the `rntis`; publishes
           a DCI 1_0's fields (for its slot)
    PDSCH  where a DCI allocated it: the TBS (38.214 5.1.3.2), LDPC (block/ldpc/nr_bg*), the transport block's CRC;
           gives the block when the CRC checks

The phase NR's upconversion gives each symbol (38.211 5.4: e^(j 2 pi f0 (t - t_start,l - N_CP,l Tc)), the carrier's phase
restarted every symbol) reaches a receiver that mixes down continuously as e^(-j 2 pi fc t_l) on symbol l: t_l its useful
part's start in the subframe, fc the absolute frequency at 0 Hz, here the SS/PBCH block's centre on the synchronization
raster. The PBCH and PDCCH, a DM-RS in every symbol, do not notice; the PDSCH, its DM-RS in some, does. Not every cell
shows it (on air, one cell of two did: one sharing its carrier with LTE did not), so the PDSCH, given `fc`, tries the block
with the phase taken off and without, the one that worked last first: the CRC says which.
"""
import functools

import numpy as np

from ..lte.channels import _Channel
from ...base.control import publish
from . import phy
from .grid import frame_len, unpack


@functools.lru_cache(maxsize=None)
def _pbch_code():
    from ... import code
    return code.load("block/polar/nr_pbch")

@functools.lru_cache(maxsize=64)
def _dci_code(K, E):
    from ... import code
    return code.load("block/polar/nr_dci", K=K, E=E)

@functools.lru_cache(maxsize=None)
def _crc(name):
    from ... import link
    return link.load(f"crc/{name}").encoder

def _interp(kq, k, h):
    return np.interp(kq, k, h.real) + 1j*np.interp(kq, k, h.imag)

def equalize(G, data, dmrs, ref):
    """Y conj(H) of the data REs (bins, symbols) from the DM-RS (bins, symbols, values `ref`): least squares at each, linear
    in frequency within a symbol, and linear in time between DM-RS symbols (constant past them); and |H|^2"""
    db, ds = data
    mb, ms = dmrs[0], dmrs[1]
    ls = G[ms, mb]/ref
    syms = np.unique(ms)
    Hs = np.array([_interp(db, mb[ms == s], ls[ms == s]) for s in syms]) # (DM-RS symbols, data REs)
    if len(syms) == 1:
        H = Hs[0]
    else:
        H = np.empty(len(db), complex)
        for j, s in enumerate(ds):
            w = np.interp(s, syms, np.arange(len(syms)))
            i0 = min(int(np.floor(w)), len(syms) - 2)
            a = w - i0
            H[j] = Hs[i0, j]*(1 - a) + Hs[i0 + 1, j]*a
    Y = G[ds, db]
    return Y*np.conj(H), np.abs(H)**2


class _NrChannel(_Channel):
    frame_len = staticmethod(frame_len)
    unpack = staticmethod(unpack)


class PBCH(_NrChannel):
    """The MIB of every SS/PBCH block NrGrid marks (see the module)"""
    def __init__(self, name="pbch"):
        super().__init__(name)

    def __str__(self):
        return "NR PBCH()"

    def frame(self, head, G, pos, meta, st):
        l0 = head["ssb_symbol"]
        if l0 < 0:
            return None
        st.est["frames"] += 1
        pci, nfft = head["pci"], head["nfft"]
        b = phy.ssb_bins(nfft)
        pk, pl, dk, dl = phy.ssb_map(pci)
        i_ssb = head["i_ssb"]
        ref = phy.pbch_dmrs(pci, i_ssb + 4*head["n_hf"])
        z, _ = equalize(G, (b[pk], l0 + pl), (b[dk], l0 + dl), ref)
        s = phy.soft_bits(z, 2)*phy.sign(phy.pbch_scramble(pci, i_ssb))
        a, err = _pbch_code().decoder.decode(s, status=True)
        if err[0] < 0:
            st.est["failed"] += 1
            return None
        mib, sfn, hrf, kmsb = phy.payload_decode(np.asarray(a, np.uint8), pci)
        m = phy.parse_mib(mib)
        fields = {"sfn": sfn, "frame_ref": head["count"]//10, "coreset0": m["coreset0"], "ss0": m["ss0"],
                  "k_ssb": m["k_ssb_lsb"] + 16*kmsb, "dmrs_pos": m["dmrs_pos"], "barred": m["barred"], "i_ssb": i_ssb}
        publish(meta, self, fields, at=pos)
        st.est["decoded"] += 1
        st.est["last"] = {**fields, "hrf": hrf}
        return mib


class PDCCH(_NrChannel):
    """DCI 1_0 in CORESET 0's Type0 occasions (see the module)"""
    CONTROLS = {k: "live" for k in ("sfn", "frame_ref", "coreset0", "ss0", "k_ssb", "i_ssb", "dmrs_pos", "barred")}

    def __init__(self, rntis=(phy.SI_RNTI,), name="pdcch"):
        super().__init__(name, rntis=tuple(int(r) for r in rntis), sfn=None, frame_ref=0, coreset0=None, ss0=None,
                         k_ssb=0, i_ssb=0, dmrs_pos=2, barred=0)

    def __str__(self):
        return f"NR PDCCH({', '.join(hex(r) for r in self.rntis)})"

    def frame(self, head, G, pos, meta, st):
        if self.sfn is None or self.coreset0 is None:
            return None
        c0 = phy.coreset0(int(self.coreset0))
        n_rb, nsym = c0["rbs"], c0["symbols"]
        slots, parity, first = phy.type0_slots(int(self.ss0), int(self.i_ssb), nsym)
        sfn = (int(self.sfn) + head["count"]//10 - int(self.frame_ref)) % 1024
        if head["slot"] not in slots or sfn % 2 != parity:
            return None
        st.est["frames"] += 1
        pci, nfft = head["pci"], head["nfft"]
        c0bin = phy.coreset0_first_bin(nfft, int(self.k_ssb), c0["offset"])
        if c0bin < 0 or c0bin + 12*n_rb > nfft: # CORESET 0 past the band the samples hold
            st.est["outside"] = st.est.get("outside", 0) + 1
            return None
        K = phy.dci_10_si_size(n_rb) + 24
        n_cce = n_rb*nsym//6
        found = []
        for L in (4, 8, 16):
            for cce in range(0, n_cce - L + 1, L):
                data, (mb, ms, mi) = phy.pdcch_res(c0bin, n_rb, nsym, first, pci, cce, L)
                ref = np.empty(len(mb), complex)
                for sym in np.unique(ms):
                    seq = phy.pdcch_dmrs(pci, head["slot"], sym, 3*n_rb)
                    ref[ms == sym] = seq[mi[ms == sym]]
                z, _ = equalize(G, data, (mb, ms), ref)
                s = phy.soft_bits(z, 2)*phy.sign(phy.gold(pci, 108*L))
                dec = _dci_code(K, 108*L).decoder
                paths, _ = dec.paths(s)
                res = dec.residue(paths)[0]
                for p, r in enumerate(res):
                    if int(r) in self.rntis:
                        bits = paths[0, p, :K - 24]
                        if not any(np.array_equal(bits, f[2]) for f in found):
                            found.append((L, cce, bits, int(r)))
                        break
        out = []
        for L, cce, bits, rnti in found:
            out.append(bits)
        if found:
            L, cce, bits, rnti = found[0]
            d = phy.parse_dci_10_si(bits, n_rb)
            d.update(rnti=rnti, aggregation=L, cce=cce)
            publish(meta, self, d, at=pos, span=frame_len(head["nfft"]))
            st.est["decoded"] += 1
            st.est["last"] = d
        return np.concatenate(out) if out else None


class PDSCH(_NrChannel):
    """The transport block a DCI 1_0 allocated (see the module)"""
    CONTROLS = {k: "live" for k in ("coreset0", "k_ssb", "dmrs_pos", "fdra", "tdra", "vrb", "mcs", "rv", "sii", "rnti",
                                    "aggregation", "cce")}

    def __init__(self, fc=0.0, name="pdsch"):
        super().__init__(name, coreset0=None, k_ssb=0, dmrs_pos=2, fdra=None, tdra=0, vrb=0, mcs=0, rv=0, sii=0,
                         rnti=phy.SI_RNTI, aggregation=None, cce=None, fc=float(fc))
        self.order = [True, False] if self.fc else [False] # Take the symbols' phase off: first what worked last

    def __str__(self):
        return "NR PDSCH()"

    def frame(self, head, G, pos, meta, st):
        if self.fdra is None or self.coreset0 is None:
            return None
        st.est["frames"] += 1
        if self.vrb:
            st.est["failed"] += 1 # (interleaved VRB-to-PRB mapping: not covered)
            return None
        pci, nfft = head["pci"], head["nfft"]
        c0 = phy.coreset0(int(self.coreset0))
        n_rb = c0["rbs"]
        c0bin = phy.coreset0_first_bin(nfft, int(self.k_ssb), c0["offset"])
        first, count = phy.riv_decode(int(self.fdra), n_rb)
        if c0bin + 12*first < 0 or c0bin + 12*(first + count) > nfft:
            st.est["outside"] = st.est.get("outside", 0) + 1
            return None
        t = phy.tdra(int(self.tdra) + 1, int(self.dmrs_pos))
        ssb = [head["ssb_symbol"]] if head["ssb_symbol"] >= 0 else []
        data, (qb, qs, qi) = phy.pdsch_res(nfft, c0bin, first, count, t["s"], t["l"], t["mapping"], int(self.dmrs_pos), ssb)
        ref = np.empty(len(qb), complex)
        for sym in np.unique(qs):
            seq = phy.pdsch_dmrs(pci, head["slot"], sym, int(qi.max()) + 1)
            ref[qs == sym] = seq[qi[qs == sym]]
        qm, rate = phy.mcs(int(self.mcs))
        nd = len(phy.dmrs_symbols(t["mapping"], t["s"], t["l"], int(self.dmrs_pos)))
        tbs = phy.tbs(12*t["l"] - 12*nd, count, qm, rate)
        E = qm*len(data[0])
        crc, ldpc = phy.ldpc_for(tbs, rate, E, int(self.rv), qm)
        scr = phy.sign(phy.gold((int(self.rnti) << 15) + pci, E))
        ok, b = False, None
        for turn in list(self.order):
            Gt = G*phy.symbol_phase(self.fc, nfft)[:, None] if turn else G
            yh, h2 = equalize(Gt, data, (qb, qs), ref)
            z = yh/np.maximum(h2, 1e-12)
            llr = phy.soft_bits(z, qm, noise=1/np.maximum(np.mean(h2), 1e-12))*scr
            b = np.asarray(ldpc.decoder.decode(llr)).astype(np.uint8)
            ok = not np.asarray(_crc(crc).field(b)).any()
            if ok:
                if turn != self.order[0]:
                    self.order.reverse()
                break
        st.est["decoded" if ok else "failed"] += 1
        st.est["last"] = {"tbs": tbs, "first": first, "count": count, "qm": qm, "crc": bool(ok),
                          "phase": bool(ok and self.order[0])}
        return b[:tbs] if ok else None
