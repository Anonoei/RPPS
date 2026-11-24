"""DVB-RCS2's burst time plan over the air (ETSI EN 301 545-2): its tables to a Plan, a Plan to its tables, the NCR clock

    b = rp.plan.builder("dvb-rcs2")
    for m in forward.take_messages():   # a dvbrcs2/forward profile's: "table" (SCT, FCT2, BCT, TBTP2), "gse" (NCR), "pls"
        b.feed(m)
    b.plan                              # rp.plan.Plan in NCR ticks, its assignments those of the TBTP2s seen
    b.clock(t)                          # the NCR at time t [s] of the forward capture (the NCR packets, fitted)

The tables come as messages of the `header/link/rcs2_table` definition (one GSE PDU each, protocol type 0x0082); the
NCR as GSE PDUs of protocol type 0x0081 (48 bits: 33-bit base, 6 reserved, 9-bit extension; NCR = 300 base + ext),
each the NCR of the first symbol of the SOF of the DVB-S2 PL frame two before the one that carries it (6.2.1.1): the
builder pairs them with the PL headers' positions ("pls" messages) and fits NCR = a + b * sample.

`tables(plan, ...)` and `ncr_packet(ncr)` make the PDUs a forward link sends (rpps.link.gse.pack puts them in data fields).
A tx type's burst format (BCT) is the linear modulation data block (TC-LM, Table 6-18) for types above 127: its
BTUs, start offset, payload size, modulation and unique word (from the tx type's `word` field: a word sequence, sent
as QPSK symbols); the turbo code's parameters are written as zeros (the profile that receives the bursts says the code).
"""
import json

import numpy as np

from . import Plan, TxType, FrameType, Sequence, Frame, register, COUNT_MOD
from .. import _config

NCR_HZ = 27e6
SCT, FCT2, BCT, TBTP2 = 0xA0, 0xAB, 0xAC, 0xAD
PROTO_NCR, PROTO_L2S = 0x0081, 0x0082
MODULATION = {"psk/bpsk": 0x00, "psk/qpsk": 0x01, "psk/8psk": 0x02, "16QAM": 0x03, "pi/2-bpsk": 0x05}

def _schema():
    from ..link.fields import Schema
    obj = _config.read(_config.find("header", "rcs2_table", "link"))
    return Schema(obj["fields"] + obj["payload_fields"], "msb", "rcs2_table")

def _bytes(bits):
    bits = np.asarray(bits, bool)
    return np.packbits(np.concatenate((bits, np.zeros(-len(bits) % 8, bool)))).tobytes()

def ncr_packet(ncr):
    """The NCR packet's content (6 bytes): 33-bit base, 6 reserved, 9-bit extension"""
    ncr = int(round(ncr))
    v = ((ncr//300) % (1 << 33)) << 15 | (0x3F << 9) | (ncr % 300)
    return v.to_bytes(6, "big")

def ncr_of(payload):
    v = int.from_bytes(bytes(payload)[:6], "big")
    return (v >> 15)*300 + (v & 0x1FF)

# ---- the burst format of a tx type (BCT, TC-LM data block)

def _lm_block(tx, symbol_rate):
    """Table 6-18 for tx types above 127"""
    from ..sequence import load as seq
    f = tx.fields
    uw = np.asarray(seq(f["word"]).bits, bool) if f.get("word") else np.zeros(0, bool)
    m = 2 # (QPSK symbols for the unique word)
    bits = []
    def put(v, n):
        bits.extend(((int(v) >> (n - 1 - i)) & 1) for i in range(n))
    put(tx.btus, 8)                                     # tx_block_size
    put(f.get("threshold", 0), 8)                       # threshold_es_n0
    put(255, 8)                                         # tx_start_offset_part1: 255 is zero
    put(0, 4)
    put(tx.offset, 20)                                  # tx_start_offset_part2 [NCR ticks]
    put(int(f.get("bits", 0))//8, 16)                   # payload_size [bytes]
    put(MODULATION.get(f.get("constellation", "psk/qpsk"), 0x01), 8)
    put(0, 8)                                           # P
    for _ in range(4):
        put(0, 4)
        put(0, 4)                                       # Q0..Q3
    put(0, 3); put(0, 5); put(0, 3); put(0, 5)          # Y_period, W_period: none (no puncturing patterns)
    put(len(uw)//m, 8)                                  # preamble_len [symbols]
    put(0, 8)                                           # postamble_len
    put(0, 4); put(0, 12); put(0, 8); put(0, 8)         # pilot_period, pilot_block_len, pilot_sum: none
    bits.extend(int(b) for b in uw)                     # the UW symbols, m bits each
    return _bytes(bits)

def _lm_parse(tx_type, data):
    b = np.unpackbits(np.frombuffer(bytes(data), np.uint8)).astype(int)
    i = 0
    def get(n):
        nonlocal i
        v = int("".join(map(str, b[i:i + n])) or "0", 2)
        i += n
        return v
    out = {"btus": get(8), "threshold": get(8)}
    p1 = get(8)
    get(4)
    out["offset"] = get(20)
    out["offset_quarter_symbols"] = 0 if p1 == 255 else p1
    if tx_type <= 127:
        out["waveform_id"] = get(8)
        return out
    out["payload_size"] = get(16)
    mod = get(8)
    out["modulation"] = {v: k for k, v in MODULATION.items()}.get(mod, mod)
    get(8)
    for _ in range(4):
        get(4); get(4)
    get(3); y = get(5); get(3); w = get(5)
    i += y + w # (puncturing patterns, a bit each: no stuffing between them here)
    i = -(-i//8)*8
    out["preamble"], out["postamble"] = get(8), get(8)
    get(4); out["pilot_period"] = get(12); out["pilot_block"] = get(8); out["pilot_sum"] = get(8)
    n = out["preamble"] + (out["pilot_block"] if out["pilot_period"] else 0) + out["postamble"]
    uw = b[i:i + 2*n]
    out["uw"] = "0x" + format(int("".join(map(str, uw)), 2), f"0{max(1, len(uw)//4)}x") if len(uw) else None
    return out

# ---- a plan to its tables

def tables(plan, network_id=1, version=0, superframes=()):
    """The table PDUs (GSE protocol 0x0082 payloads) of a plan in NCR ticks: SCT, FCT2, BCT, and a TBTP2 for each
    (sequence, superframe count) in `superframes` that the plan has assignments for"""
    s = _schema()
    head = {"interactive_network_id": network_id, "version_number": version % 32, "current_next_indicator": 1}
    sct = {"superframe_loop_count": len(plan.sequences) - 1, "superframes": [
        {"superframe_sequence": q.id, "superframe_start_time_base": (q.start//300) % (1 << 33), "superframe_start_time_ext": q.start % 300,
         "superframe_duration": q.duration, "superframe_centre_frequency": int(round(q.centre/100)) % (1 << 32),
         "superframe_count": q.count % COUNT_MOD, "frame_loop_count": len(q.frames) - 1,
         "frames": [{"frame_type": f.type, "frame_start_time": f.start, "frame_centre_frequency_offset": int(round(f.offset/100))}
                    for f in q.frames]} for q in plan.sequences.values()]}
    fct = {"frame_type_loop_count": len(plan.frame_types) - 1, "frame_types": [
        {"frame_type": ft.id, "frame_duration": ft.duration, "tx_format_class": 1, "btu_duration": ft.btu_duration,
         "btu_carrier_bw": int(round(ft.btu_bw/10)), "btu_symbol_rate": int(round(ft.symbol_rate/10)), "time_unit_count": ft.time_units,
         "grid_repeat_count": len(ft.grid), "grid_frequency_offsets": [{"grid_frequency_offset": int(round(g/100))} for g in ft.grid],
         "section_loop_count": len(ft.sections) - 1,
         "sections": [{"default_tx_type": t, "fixed_access_method": 0, "repeat_count": r} for t, r in ft.sections]}
        for ft in plan.frame_types.values()]}
    rate = {t: ft.symbol_rate for ft in plan.frame_types.values() for t, _ in ft.sections}
    bct = {"tx_type_loop_count": len(plan.tx_types), "tx_types": [
        {"tx_type": tx.id, "tx_content_type": 1, "tx_format_class": 1, "tx_format_data_length": len(d), "tx_format_data": d}
        for tx in plan.tx_types.values() for d in [_lm_block(tx, rate.get(tx.id))]]}
    out = [_bytes(s.build({**head, "table_id": SCT, "sct": sct})), _bytes(s.build({**head, "table_id": FCT2, "fct2": fct})),
           _bytes(s.build({**head, "table_id": BCT, "bct": bct}))]
    for seq, count in superframes:
        given = plan.assignments.get((seq, count % COUNT_MOD))
        if not given:
            continue
        af = 11
        frames = [{"frame_number": f, "assignment_offset": off, "assignment_loop_count": len(items) - 1,
                   "assignments": [{"assignment_af11": {"dynamic_tx_type": tx, "assignment_id": who}} for who, tx in items]}
                  for f, (off, items) in sorted(given.items())]
        t = {"group_id": 0, "superframe_sequence": seq, "assignment_context": 0, "superframe_count": count % 256,
             "assignment_format": af, "frame_loop_count": len(frames) - 1, "frames": frames}
        out.append(_bytes(s.build({**head, "table_id": TBTP2, "tbtp2": t})))
    return out


@register("dvb-rcs2")
class Builder:
    """DVB-RCS2's plan from a forward link's messages (see the module). `plan` is None until SCT, FCT2 and BCT are in"""
    def __init__(self):
        self.sct = self.fct2 = self.bct = None
        self.tbtp2 = []
        self.sofs, self.ncrs = [], [] # (sample) of PL frame starts; (sample of the packet, NCR)
        self._plan, self._fit = None, None

    def feed(self, m):
        f = m.fields
        if m.kind == "pls" or "modcod" in f and m.kind != "table": # A PL header: a PL frame's start
            self.sofs.append(m.sample)
        elif m.kind == "gse" and f.get("protocol") == PROTO_NCR and m.payload:
            self.ncrs.append((m.sample, ncr_of(m.payload)))
            self._fit = None
        elif "table_id" in f and m.crc_ok is not False:
            p = f.get("payload") or {}
            tid = f["table_id"]
            if tid == SCT and "sct" in p:
                self.sct, self._plan = p["sct"], None
            elif tid == FCT2 and "fct2" in p:
                self.fct2, self._plan = p["fct2"], None
            elif tid == BCT and "bct" in p:
                self.bct, self._plan = p["bct"], None
            elif tid == TBTP2 and "tbtp2" in p:
                self.tbtp2.append(p["tbtp2"])
                if self._plan is not None:
                    self._assign(self._plan, p["tbtp2"])

    # ---- the NCR

    def pairs(self):
        """(sample of the reference SOF, NCR) of each NCR packet: the SOF two PL frames before the one carrying it"""
        sofs = np.sort(np.asarray(self.sofs, dtype=np.float64))
        out = []
        for s, ncr in self.ncrs:
            k = int(np.searchsorted(sofs, s, side="right")) - 1 # the PL frame whose data carries the packet
            if k - 2 >= 0:
                out.append((sofs[k - 2], ncr))
        return out

    def ncr_fit(self):
        """(a, b): NCR = a + b * forward sample, least squares over the pairs (the 2^42 wrap of the NCR undone)"""
        if self._fit is None:
            pr = self.pairs()
            if len(pr) < 2:
                return None
            x = np.array([p[0] for p in pr])
            y = np.unwrap(np.array([p[1] for p in pr], dtype=np.float64), period=float(1 << 33)*300)
            b, a = np.polyfit(x, y, 1)
            self._fit = (a, b)
        return self._fit

    def clock(self, sample):
        """The NCR at a forward capture sample (a time the return capture shares: its first sample's NCR is the plan's
        origin there)"""
        fit = self.ncr_fit()
        return None if fit is None else fit[0] + fit[1]*sample

    # ---- the plan

    @property
    def plan(self):
        if self._plan is None and self.sct and self.fct2 and self.bct:
            self._plan = self._build()
            for t in self.tbtp2:
                self._assign(self._plan, t)
        return self._plan

    def _build(self):
        tx = {}
        for t in self.bct["tx_types"]:
            d = _lm_parse(t["tx_type"], t["tx_format_data"])
            fields = {k: v for k, v in d.items() if k not in ("btus", "offset")}
            tx[t["tx_type"]] = TxType(t["tx_type"], d["btus"], d["offset"], fields)
        fts = {}
        for f in self.fct2["frame_types"]:
            grid = [g["grid_frequency_offset"]*100.0 for g in f.get("grid_frequency_offsets", [])] or [0.0]
            fts[f["frame_type"]] = FrameType(f["frame_type"], f["frame_duration"], f["btu_duration"], f["btu_carrier_bw"]*10.0,
                                             f["btu_symbol_rate"]*10.0, f["time_unit_count"], grid,
                                             [(s["default_tx_type"], s["repeat_count"]) for s in f["sections"]])
        seqs = {}
        for s in self.sct["superframes"]:
            start = s["superframe_start_time_base"]*300 + s["superframe_start_time_ext"]
            seqs[s["superframe_sequence"]] = Sequence(s["superframe_sequence"], start, s["superframe_duration"],
                                                      s["superframe_centre_frequency"]*100.0, s["superframe_count"],
                                                      [Frame(fr["frame_type"], fr["frame_start_time"], fr["frame_centre_frequency_offset"]*100.0)
                                                       for fr in s["frames"]])
        return Plan(NCR_HZ, tx, fts, seqs, name="dvb-rcs2")

    def _assign(self, plan, t):
        seq = plan.sequences.get(t["superframe_sequence"])
        if seq is None:
            return
        count = seq.count + (t["superframe_count"] - seq.count) % 256 # (8 bits in the TBTP2: the SCT's count, onwards)
        for f in t["frames"]:
            items = []
            for a in f["assignments"]:
                v = next(iter(a.values()))
                items.append((v["assignment_id"], v.get("dynamic_tx_type", 0)))
            plan.assign(seq.id, count, f["frame_number"], items, f["assignment_offset"])
