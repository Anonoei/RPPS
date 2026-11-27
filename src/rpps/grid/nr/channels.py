"""NR downlink channels read off NrGrid's frames: PBCH (MIB), PDCCH (DCI 1_0 in CORESET 0), PDSCH (SIB1)

The PBCH is at the SS/PBCH block's spacing, the PDCCH and PDSCH at the one its MIB names (15 or 30 kHz each). Each
estimates its channel from its own DM-RS (least squares, linear between them in frequency, and in time for the
PDSCH) and publishes what it decoded from the first item of its slot (rpps.base.control.publish), as rpps.grid.lte's do:

    PBCH   subframes NrGrid marks as holding the block: the MIB and the payload's timing (SFN, half frame, k_SSB's MSB);
           publishes sfn (and the output frame it was read in, frame_ref), coreset0, ss0, k_ssb, dmrs_pos, barred
    PDCCH  once the MIB is known, and unless its k_SSB says the block has no SIB1 (above 23: est "no_sib1"): the Type0 occasions (38.213 13: slots n0 and the one after, of frames of the right
           parity) of CORESET 0 (`min_bw`: the band's minimum channel bandwidth, which picks its table), every candidate of aggregation 4, 8 and 16 (polar, block/polar/nr_dci), for the `rntis`; publishes
           a DCI 1_0's fields (for its subframe, with its slot)
    PDSCH  where a DCI allocated it (contiguous virtual blocks, interleaved onto the CORESET's or not): the TBS (38.214 5.1.3.2), LDPC (block/ldpc/nr_bg*), the transport block's CRC;
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
from ...base.control import publish, emit
from . import phy
from .grid import frame_len, unpack


@functools.lru_cache(maxsize=64)
def _dci_code(K, E):
    from ... import code
    return code.load("block/polar/nr_dci", K=K, E=E)

@functools.lru_cache(maxsize=None)
def _crc(name):
    from ... import link
    return link.load(f"crc/{name}").encoder

class _NrChannel(_Channel):
    frame_len = staticmethod(frame_len)
    unpack = staticmethod(unpack)


class PBCH(_NrChannel):
    """The MIB of every SS/PBCH block NrGrid marks (see the module)"""
    check_name = "crc24_nr_c" # (Message.check: the polar code's CRC-24C)
    def __init__(self, name="pbch"):
        super().__init__(name)

    def __str__(self):
        return "NR PBCH()"

    def frame(self, head, G, B, pos, meta, st):
        if head["ssb_symbol"] < 0:
            return None
        st.est["frames"] += 1
        pci = head["pci"]
        got = phy.read_pbch(B, pci)
        if got is None:
            st.est["failed"] += 1
            emit(meta, self, None, {"pci": pci, "i_ssb": head["i_ssb"]}, None, False, pos)
            return None
        mib = got["mib"]
        m = phy.parse_mib(mib)
        fields = {"sfn": got["sfn"], "frame_ref": head["count"]//10, "coreset0": m["coreset0"], "ss0": m["ss0"],
                  "k_ssb": m["k_ssb_lsb"] + 16*got["k_ssb_msb"], "dmrs_pos": m["dmrs_pos"], "barred": m["barred"],
                  "i_ssb": got["i_ssb"]}
        publish(meta, self, fields, at=pos)
        extra = {"hrf": got["hrf"], "scs": 15e3*(1 << head["mu_ssb"]), "scs_common": 15e3*(1 << head["mu"])}
        emit(meta, self, None, {"pci": pci, **fields, **extra}, mib, True, pos)
        st.est["decoded"] += 1
        st.est["last"] = {**fields, **extra}
        return mib


class PDCCH(_NrChannel):
    """DCI 1_0 in CORESET 0's Type0 occasions (see the module)"""
    check_name = "crc24_nr_c" # (Message.check: the polar code's CRC-24C, scrambled with the RNTI)
    CONTROLS = {k: "live" for k in ("sfn", "frame_ref", "coreset0", "ss0", "k_ssb", "i_ssb", "dmrs_pos", "barred")}

    def __init__(self, rntis=(phy.SI_RNTI,), min_bw=5, name="pdcch"):
        super().__init__(name, rntis=tuple(int(r) for r in rntis), min_bw=int(min_bw), sfn=None, frame_ref=0,
                         coreset0=None, ss0=None, k_ssb=0, i_ssb=0, dmrs_pos=2, barred=0)

    def __str__(self):
        return f"NR PDCCH({', '.join(hex(r) for r in self.rntis)})"

    def frame(self, head, G, B, pos, meta, st):
        if self.sfn is None or self.coreset0 is None:
            return None
        if int(self.k_ssb) > phy.SI_CORESET_MAX_K_SSB: # No SIB1 with this block (a non-standalone cell's, say)
            st.est["no_sib1"] = st.est.get("no_sib1", 0) + 1
            return None
        mu = head["mu"]
        scs, scs_common = 15e3*(1 << head["mu_ssb"]), 15e3*(1 << mu)
        c0 = phy.coreset0(int(self.coreset0), scs, scs_common, self.min_bw)
        n_rb, nsym = c0["rbs"], c0["symbols"]
        (n0, _), parity, first = phy.type0_slots(int(self.ss0), int(self.i_ssb), nsym, mu)
        sfn = (int(self.sfn) + head["count"]//10 - int(self.frame_ref)) % 1024
        per_frame = 10 << mu
        found, here = [], None
        for s in range(1 << mu):
            slot = (head["subframe"] << mu) + s
            # An occasion is slot n0 of a frame of the parity, or the one after it (the next frame's first, past the last)
            if not any((slot + j*per_frame - d) == n0 and (sfn - j) % 2 == parity for d in (0, 1) for j in (0, 1)):
                continue
            st.est["frames"] += 1
            c0bin = phy.coreset0_first_bin(head["nfft"], int(self.k_ssb), c0["offset"], scs, scs_common)
            if c0bin is None or c0bin < 0 or c0bin + 12*n_rb > head["nfft"]: # CORESET 0 past the band the samples hold
                st.est["outside"] = st.est.get("outside", 0) + 1
                continue
            got = self.search(head["pci"], G[s], slot, c0bin, n_rb, nsym, first)
            if got and here is None:
                here = (slot, got[0])
            for i, (L, cce, bits, rnti) in enumerate(got):
                found.append(bits)
                fields = {"pci": head["pci"], "slot": slot, "sfn": sfn, "rnti": rnti, "aggregation": L, "cce": cce}
                if i == 0:
                    fields.update(phy.parse_dci_10_si(bits, n_rb))
                emit(meta, self, None, fields, bits, True, pos) # (its CRC, scrambled with the RNTI, matched)
        if here: # (one PDSCH a subframe is received: the first DCI's)
            slot, (L, cce, bits, rnti) = here
            d = phy.parse_dci_10_si(bits, n_rb)
            d.update(rnti=rnti, aggregation=L, cce=cce, slot=slot)
            publish(meta, self, d, at=pos, span=frame_len(head["nfft"] << mu))
            st.est["decoded"] += 1
            st.est["last"] = d
        return np.concatenate(found) if found else None

    def search(self, pci, G, slot, c0bin, n_rb, nsym, first):
        """(aggregation, CCE, payload bits, RNTI) of every DCI for the RNTIs in one slot's grid"""
        K = phy.dci_10_si_size(n_rb) + 24
        n_cce = n_rb*nsym//6
        found = []
        for L in (4, 8, 16): # Every candidate of a level in one list decode (they share K and E)
            starts = list(range(0, n_cce - L + 1, L))
            if not starts:
                continue
            soft = []
            for cce in starts:
                data, (mb, ms, mi) = phy.pdcch_res(c0bin, n_rb, nsym, first, pci, cce, L)
                ref = np.empty(len(mb), complex)
                for sym in np.unique(ms):
                    seq = phy.pdcch_dmrs(pci, slot, sym, 3*n_rb)
                    ref[ms == sym] = seq[mi[ms == sym]]
                z, _ = phy.equalize(G, data, (mb, ms), ref)
                soft.append(phy.soft_bits(z, 2)*phy.sign(phy.gold(pci, 108*L)))
            dec = _dci_code(K, 108*L).decoder
            paths, _ = dec.paths(np.concatenate(soft))
            res = dec.residue(paths)
            for w, cce in enumerate(starts):
                for p, r in enumerate(res[w]):
                    if int(r) in self.rntis:
                        bits = paths[w, p, :K - 24]
                        if not any(np.array_equal(bits, f[2]) for f in found):
                            found.append((L, cce, bits, int(r)))
                        break
        return found


class PDSCH(_NrChannel):
    """The transport block a DCI 1_0 allocated (see the module)"""
    check_name = None # (Message.check: the transport block's CRC, by its size: set as it is decoded)
    CONTROLS = {k: "live" for k in ("coreset0", "k_ssb", "dmrs_pos", "fdra", "tdra", "vrb", "mcs", "rv", "sii", "rnti",
                                    "aggregation", "cce", "slot")}

    def __init__(self, fc=0.0, min_bw=5, name="pdsch"):
        super().__init__(name, coreset0=None, k_ssb=0, dmrs_pos=2, fdra=None, tdra=0, vrb=0, mcs=0, rv=0, sii=0,
                         rnti=phy.SI_RNTI, aggregation=None, cce=None, slot=None, fc=float(fc), min_bw=int(min_bw))
        self.order = [True, False] if self.fc else [False] # Take the symbols' phase off: first what worked last

    def __str__(self):
        return "NR PDSCH()"

    def frame(self, head, G, B, pos, meta, st):
        if self.fdra is None or self.coreset0 is None or self.slot is None:
            return None
        st.est["frames"] += 1
        mu, pci, nfft, slot = head["mu"], head["pci"], head["nfft"], int(self.slot)
        s_in = slot % (1 << mu)
        scs, scs_common = 15e3*(1 << head["mu_ssb"]), 15e3*(1 << mu)
        c0 = phy.coreset0(int(self.coreset0), scs, scs_common, self.min_bw)
        n_rb = c0["rbs"]
        c0bin = phy.coreset0_first_bin(nfft, int(self.k_ssb), c0["offset"], scs, scs_common)
        first, count = phy.riv_decode(int(self.fdra), n_rb)
        if first < 0 or count < 1 or first + count > n_rb:
            st.est["failed"] += 1
            return None
        prbs = np.arange(first, first + count)
        if self.vrb:
            prbs = phy.vrb_interleaver(n_rb)[prbs]
        if c0bin is None or c0bin + 12*int(prbs.min()) < 0 or c0bin + 12*(int(prbs.max()) + 1) > nfft:
            st.est["outside"] = st.est.get("outside", 0) + 1
            return None
        t = phy.tdra(int(self.tdra) + 1, int(self.dmrs_pos))
        blocks = []
        if head["ssb_symbol"] >= 0 and head["ssb_symbol"] < 14*(s_in + 1) and head["ssb_end"] > 14*s_in:
            half = 120*(1 << head["mu_ssb"])//(1 << mu) # the block's subcarriers either side, in this grid's
            blocks.append((head["ssb_symbol"] - 14*s_in, head["ssb_end"] - 14*s_in, nfft//2 - half, nfft//2 + half))
        data, (qb, qs, qi) = phy.pdsch_res(c0bin, prbs, t["s"], t["l"], t["mapping"], int(self.dmrs_pos), blocks)
        ref = np.empty(len(qb), complex)
        for sym in np.unique(qs):
            seq = phy.pdsch_dmrs(pci, slot, sym, int(qi.max()) + 1)
            ref[qs == sym] = seq[qi[qs == sym]]
        qm, rate = phy.mcs(int(self.mcs))
        nd = len(phy.dmrs_symbols(t["mapping"], t["s"], t["l"], int(self.dmrs_pos)))
        tbs = phy.tbs(12*t["l"] - 12*nd, count, qm, rate)
        E = qm*len(data[0])
        crc, ldpc = phy.ldpc_for(tbs, rate, E, int(self.rv), qm)
        self.check_name = crc
        scr = phy.sign(phy.gold((int(self.rnti) << 15) + pci, E))
        ok, b = False, None
        for turn in list(self.order):
            Gt = G[s_in]*phy.symbol_phase(self.fc, nfft, mu, slot)[:, None] if turn else G[s_in]
            yh, h2 = phy.equalize(Gt, data, (qb, qs), ref)
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
        emit(meta, self, None, {"pci": pci, "slot": slot, "rnti": int(self.rnti), "tbs": tbs, "first": first,
                                "count": count, "vrb": int(self.vrb), "mcs": int(self.mcs), "qm": qm, "rv": int(self.rv)},
             b[:tbs] if ok else None, bool(ok), pos)
        return b[:tbs] if ok else None
