"""LTE downlink physical channels read off LteGrid's frames: PBCH (MIB), PCFICH (CFI), PDCCH (DCI), PDSCH (transport blocks)

Each reads whole subframe frames (rpps.grid.lte.grid: a partial frame waits for the rest) and gives bits. Each publishes
what it decoded from the first item of its subframe (rpps.base.control.publish), so a profile's controls carry it to the
channels after it: the MIB's port count to every channel, the CFI to the PDCCH and PDSCH of its subframe, a DCI's
allocation to the PDSCH of its subframe (3GPP TS 36.211, 36.212, 36.213). Coding is rpps's: the LTE Gold sequence,
tail-biting convolutional code with its rate matching, turbo code with its rate matching, CRC-16 and CRC-24A.

    PBCH   subframe 0: 1, 2 or 4 ports and the frame's quarter of the 40 ms scrambling are tried, the CRC mask says which:
           gives the 24 MIB bits; publishes ports, n_rb, phich_ng, phich_duration, sfn
    PCFICH every subframe, once `ports` is known: gives the CFI (2 bits); publishes cfi (for its subframe)
    PDCCH  once ports, cfi and phich_ng are known: the common search space (CCEs 0-15, aggregation 4 and 8) for the
           `rntis`, formats 1A and 1C: gives each DCI's bits; publishes a 1A's fields (for its subframe)
    PDSCH  where a DCI 1A gave an allocation (riv not None): localized, one code block; gives the transport block when its
           CRC-24A checks
"""
import functools
from types import SimpleNamespace

import numpy as np

from ...base.rpps import Pipe
from ...base.types import Category, Rep, dtype
from ...base.data import Digital
from ...base.control import publish, emit
from . import phy
from .grid import unpack, frame_len


@functools.lru_cache(maxsize=None)
def _conv():
    from ... import code
    return code.Conv((0o133, 0o171, 0o165), 7, termination="tailbiting") # 36.212 5.1.3.1

@functools.lru_cache(maxsize=256)
def _conv_rm(D, E):
    from ... import code
    return code.load("block/ratematch/lte_conv", D=D, E=E).encoder

@functools.lru_cache(maxsize=None)
def _crc(name):
    from ... import link
    return link.load(f"crc/{name}").encoder

@functools.lru_cache(maxsize=64)
def _turbo(K):
    from ... import code
    return code.load("block/turbo/lte", K=K).encoder

@functools.lru_cache(maxsize=256)
def _turbo_rm(K, E, rv):
    from ... import code
    return code.load("block/ratematch/lte_turbo", K=K, E=E, rv=rv).encoder

@functools.lru_cache(maxsize=None)
def tbs_table():
    """{I_TBS: [TBS for N_PRB 1..10]} (36.213 Table 7.1.7.2.1-1, the columns a DCI 1A for system information uses)"""
    from ... import _config
    t = _config.read(_config.find("table", "lte/tbs"))
    return {r[0]: list(r[1:]) for r in t["rows"]}

def tail_biting(llr, K):
    """K bits (CRC included) from soft bits of the 1/3 tail-biting code, rate matched to len(llr) (positive = 0)"""
    d = _conv_rm(K, len(llr)).decode(np.asarray(llr, float))
    return np.asarray(_conv().decoder.decode(d)).astype(np.uint8)

def crc_bits(bits, name):
    return np.asarray(_crc(name).field(np.asarray(bits, np.uint8))).astype(np.uint8)

def _int(bits):
    v = 0
    for b in bits:
        v = (v << 1) | int(b)
    return v


class _Channel(Pipe):
    """A channel on grid frames: whole frames are handed to `frame(head, *planes, pos, meta, st)`, which gives bits. The
    frame layout: `frame_len(head item 2)` items a frame, `unpack(frame)` its header and planes (LTE's here; NR's in
    rpps.grid.nr)"""
    CATEGORY = Category.SYMBOL
    CONTROLS = {}
    EMITS = True # Every decode is a message (rpps.profile.message)
    frame_len = staticmethod(frame_len)
    unpack = staticmethod(unpack)

    def __init__(self, name, **params):
        self.name = name
        for k, v in params.items():
            setattr(self, k, v)

    def takes(self, encode=False):
        return {Rep.SYMBOLS}

    def control(self, **params):
        for k, v in params.items():
            if k not in self.CONTROLS:
                raise ValueError(f"{self.name}: no parameter {k!r} to control")
            setattr(self, k, v)

    def new_state(self, meta):
        return SimpleNamespace(held=None, F=None, est={"frames": 0, "decoded": 0, "failed": 0, "last": None})

    def __radd__(self, data):
        from ...base.data import _Data, expect
        if not isinstance(data, _Data):
            return NotImplemented
        expect(self, data, self.takes(), True)
        meta = data.meta
        st = meta.state_of(self)
        x = np.asarray(data.data)
        pos0 = meta.pos
        if st.held is not None and len(st.held):
            pos0 -= len(st.held)
            x = np.concatenate((st.held, x))
        if st.F is None and len(x) >= 4:
            st.F = self.frame_len(int(round(x[2].real)))
        out = []
        n = 0 if st.F is None else len(x)//st.F
        for i in range(n):
            f = x[i*st.F:(i + 1)*st.F]
            head, *planes = self.unpack(f)
            if head["pci"] < 0:
                continue
            bits = self.frame(head, *planes, pos0 + i*st.F, meta, st)
            if bits is not None:
                out.append(np.asarray(bits, bool))
        st.held = x[n*st.F:] if st.F else x
        meta.stream.est[self.name] = dict(st.est)
        y = np.concatenate(out) if out else np.zeros(0, bool)
        stream = meta.stream
        om = meta.fork()
        om.pos = stream.out_pos.get(self, 0)
        stream.out_pos[self] = om.pos + len(y)
        return data.derive(y, self.CATEGORY, dtype.BITS, Digital, dict(data.ctx), meta=om, rep=Rep.DATA)

    def frame(self, head, G, H, pos, meta, st):
        raise NotImplementedError


class PBCH(_Channel):
    """The MIB of every subframe 0 (see the module)"""
    def __init__(self, name="pbch"):
        super().__init__(name)

    def __str__(self):
        return "PBCH()"

    def frame(self, head, G, H, pos, meta, st):
        if head["sf"] != 0:
            return None
        st.est["frames"] += 1
        pci, n_rb = head["pci"], head["n_rb"]
        k, l = phy.pbch_res(pci, n_rb)
        c = phy.sign(phy.gold(pci, 1920))
        for ports in (1, 2, 4):
            s = phy.soft_bits(phy.undo_diversity(G, H, k, l, ports))
            for q in range(4):
                llr = np.zeros(1920)
                llr[q*480:(q + 1)*480] = s*c[q*480:(q + 1)*480]
                b = tail_biting(llr, 40)
                mask = b[24:] ^ crc_bits(b[:24], "crc16_xmodem")
                if np.array_equal(mask, {1: [0]*16, 2: [1]*16, 4: [0, 1]*8}[ports]): # 36.212 Table 5.3.1.1-1
                    mib = phy.parse_mib(b[:24])
                    fields = {"ports": ports, "n_rb": mib["n_rb"], "phich_ng": mib["phich_ng"],
                              "phich_duration": mib["phich_duration"], "sfn": mib["sfn_msb"]*4 + q}
                    publish(meta, self, fields, at=pos)
                    emit(meta, self, None, {"pci": pci, "sf": 0, **fields}, b[:24], True, pos)
                    st.est["decoded"] += 1
                    st.est["last"] = fields
                    return b[:24]
        st.est["failed"] += 1
        emit(meta, self, None, {"pci": pci, "sf": 0}, None, False, pos)
        return None


class PCFICH(_Channel):
    """The CFI of every subframe, once the port count is known (see the module)"""
    CONTROLS = {"ports": "live"}

    def __init__(self, ports=None, name="pcfich"):
        super().__init__(name, ports=ports)

    def __str__(self):
        return f"PCFICH({self.ports} ports)"

    def frame(self, head, G, H, pos, meta, st):
        if self.ports is None:
            return None
        st.est["frames"] += 1
        pci, sf = head["pci"], head["sf"]
        k, l = phy.pcfich_res(pci, head["n_rb"], self.ports)
        s = phy.soft_bits(phy.undo_diversity(G, H, k, l, self.ports))
        s = s*phy.sign(phy.gold(((sf + 1)*(2*pci + 1) << 9) + pci, 32))
        score = {cfi: float(np.sum(s*phy.sign(w))) for cfi, w in phy.CFI_WORDS.items()}
        cfi = max(score, key=score.get)
        publish(meta, self, {"cfi": cfi}, at=pos, span=frame_len(head["n_rb"]))
        emit(meta, self, None, {"pci": pci, "sf": sf, "cfi": cfi}, None, None, pos) # (no check: the nearest code word)
        st.est["decoded"] += 1
        st.est["last"] = {"cfi": cfi}
        return [cfi >> 1 & 1, cfi & 1]


class PDCCH(_Channel):
    """DCIs in the common search space for the RNTIs given (see the module)"""
    CONTROLS = {"ports": "live", "cfi": "live", "phich_ng": "live"}

    def __init__(self, rntis=(phy.SI_RNTI,), ports=None, cfi=None, phich_ng=None, name="pdcch"):
        super().__init__(name, rntis=tuple(int(r) for r in rntis), ports=ports, cfi=cfi, phich_ng=phich_ng)

    def __str__(self):
        return f"PDCCH({', '.join(hex(r) for r in self.rntis)})"

    def cces(self, head, G, H):
        """(N_CCE, 72) soft bits of the control region: REGs equalized, the shift and interleaver undone, descrambled"""
        pci, n_rb, sf = head["pci"], head["n_rb"], head["sf"]
        key = (pci, n_rb, phy.control_symbols(self.cfi, n_rb), phy.phich_groups(self.phich_ng, n_rb), int(self.ports))
        M = len(phy.control_regs(*key))
        k, l = phy.control_res(*key)
        wbar = phy.soft_bits(phy.undo_diversity(G, H, k, l, int(self.ports))).reshape(M, 8)
        w = wbar[(np.arange(M) - pci) % M] # Undo the cyclic shift by the cell id
        z = np.zeros_like(w)
        z[np.asarray(phy.subblock_perm(M))] = w # and the quadruplet interleaver
        bits = z.reshape(-1)*phy.sign(phy.gold((sf << 9) + pci, 8*M))
        n = M//9
        return bits[:n*72].reshape(n, 72)

    def frame(self, head, G, H, pos, meta, st):
        if None in (self.ports, self.cfi, self.phich_ng):
            return None
        st.est["frames"] += 1
        n_rb = head["n_rb"]
        cc = self.cces(head, G, H)
        sizes = {phy.dci_1a_size(n_rb): "1A", phy.dci_1c_size(n_rb): "1C"}
        found, seen = [], set()
        for L, starts in ((4, (0, 4, 8, 12)), (8, (0, 8))):
            for s0 in starts:
                if s0 + L > len(cc):
                    continue
                llr = cc[s0:s0 + L].reshape(-1)
                for K, fmt in sizes.items():
                    b = tail_biting(llr, K + 16)
                    rnti = _int(b[K:] ^ crc_bits(b[:K], "crc16_xmodem"))
                    key = (rnti, K, b[:K].tobytes())
                    if rnti in self.rntis and key not in seen: # (aggregation 8 holds the 4 of the same DCI)
                        seen.add(key)
                        found.append((fmt, rnti, b[:K]))
        out, allocated = [], False
        for fmt, rnti, b in found:
            out.append(b)
            fields = {"pci": head["pci"], "sf": head["sf"], "format": fmt, "rnti": rnti}
            if fmt == "1A" and b[0] == 1:
                fields.update(phy.parse_1a(b, n_rb))
            emit(meta, self, None, fields, b, True, pos) # (its CRC, masked with the RNTI, matched)
            if fmt == "1A" and b[0] == 1 and not allocated: # (the first 1A allocates: one PDSCH a subframe here)
                allocated = True
                d = phy.parse_1a(b, n_rb)
                d["rnti"] = rnti
                publish(meta, self, d, at=pos, span=frame_len(n_rb))
                st.est["last"] = d
        st.est["decoded"] += len(found)
        return np.concatenate(out) if out else None


class PDSCH(_Channel):
    """The transport block a DCI 1A allocated in its subframe (see the module)"""
    CONTROLS = {k: "live" for k in ("ports", "cfi", "rnti", "riv", "mcs", "rv", "tpc", "distributed", "flag", "harq", "ndi")}

    def __init__(self, ports=None, cfi=None, name="pdsch"):
        super().__init__(name, ports=ports, cfi=cfi, rnti=None, riv=None, mcs=None, rv=0, tpc=0, distributed=0, flag=1,
                         harq=0, ndi=0)

    def __str__(self):
        return "PDSCH()"

    def frame(self, head, G, H, pos, meta, st):
        if None in (self.ports, self.cfi, self.riv, self.rnti) or self.distributed or not self.flag:
            return None
        st.est["frames"] += 1
        pci, n_rb, sf = head["pci"], head["n_rb"], head["sf"]
        first, count = phy.riv_decode(int(self.riv), n_rb)
        tbs = tbs_table().get(int(self.mcs))
        if tbs is None:
            st.est["failed"] += 1
            return None
        tbs = tbs[(3 if int(self.tpc) & 1 else 2) - 1] # N_PRB^1A from the TPC's low bit (36.213 7.1.7)
        k, l = phy.pdsch_res(first, count, phy.control_symbols(self.cfi, n_rb), pci, sf, n_rb, int(self.ports))
        s = phy.soft_bits(phy.undo_diversity(G, H, k, l, int(self.ports)))
        s = s*phy.sign(phy.gold((int(self.rnti) << 14) + (sf << 9) + pci, len(s)))
        K = tbs + 24
        t = _turbo(K)
        llr = t.app(_turbo_rm(K, len(s), int(self.rv)).decode(s))
        b = (np.asarray(llr) < 0).astype(np.uint8)
        ok = not crc_bits(b, "crc24_lte_a").any()
        st.est["decoded" if ok else "failed"] += 1
        st.est["last"] = {"tbs": tbs, "first": first, "count": count, "crc": bool(ok)}
        emit(meta, self, None, {"pci": pci, "sf": sf, "rnti": int(self.rnti), "tbs": tbs, "first": first, "count": count,
                                "mcs": int(self.mcs), "rv": int(self.rv)}, b[:tbs] if ok else None, bool(ok), pos)
        return b[:tbs] if ok else None
