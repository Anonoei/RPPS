"""A burst time plan over the air (DVB-RCS2): GSE in BBFRAMEs (rpps.link.gse), the RCS2 tables (header/link/rcs2_table)
and the NCR (header/link/rcs2_ncr), plan/dvbrcs2/forward's builder; a whole synthetic network: the forward link tells the plan, the return link's
bursts are found and named by it"""
import numpy as np
import pytest

import rpps as rp
from rpps.base.types import Rep
from rpps.link import gse

def _gse_rx(fields, protocols=(130,), cuts=1):
    p = rp._config.load_obj("profile", "test/gse", {"base": "profile", "steps": [{"link": "gse", "id": "gse", "protocols": list(protocols)}]})
    rx, msgs, out = p.rx(), [], []
    op = rx.slots[0].op
    for i, f in enumerate(fields):
        op.control(frame=1) # (a BBHEADER's frame start)
        bits = np.unpackbits(np.frombuffer(f, np.uint8)).astype(bool)
        for k, part in enumerate(np.array_split(bits, cuts)):
            d = rp.Data(part, flush=(i == len(fields) - 1 and k == cuts - 1))
            out.append(np.asarray(rx(d.derive(d.data, d.CAT, d.DT, rep=Rep.DATA)).data))
            msgs += rx.take_messages()
    return msgs, np.concatenate(out)

@pytest.mark.parametrize("cuts", [1, 7])
def test_gse_fragments_and_crc(rng, cuts):
    pdus = [(0x0082, bytes(rng.integers(0, 256, n).astype(np.uint8))) for n in (40, 300, 1500, 12)] + [(0x0081, bytes(range(6)))]
    fields = gse.pack(pdus, 400)
    assert all(len(f) == 400 for f in fields) and len(fields) == 5
    msgs, out = _gse_rx(fields, cuts=cuts)
    got = sorted((m.fields["protocol"], m.payload) for m in msgs if m.kind == "gse" and m.crc_ok)
    assert got == sorted((p, d) for p, d in pdus)
    # Only the PDUs of protocol 0x0082 come out, each whole, in the order they end (packed in order: the 12-byte one after
    # the last fragment of the 1500-byte one)
    assert np.packbits(out).tobytes() == b"".join(d for p, d in pdus if p == 0x0082)

def test_gse_bad_fragment_is_a_failed_message(rng):
    fields = gse.pack([(0x0082, bytes(rng.integers(0, 256, 900).astype(np.uint8)))], 400)
    bad = bytearray(fields[1])
    bad[50] ^= 0x10
    msgs, out = _gse_rx([fields[0], bytes(bad), fields[2]])
    assert [m.crc_ok for m in msgs if m.kind == "gse"] == [False] and len(out) == 0

def _table_message(pdu):
    """A table PDU as the forward profile's header step gives it: a message of its fields, the content under payload"""
    from rpps.link.fields import Schema
    obj = rp._config.read(rp._config.find("header", "rcs2_table", "link"))
    v, _ = Schema(obj["fields"] + obj["payload_fields"], "msb", "test").parse(np.unpackbits(np.frombuffer(pdu, np.uint8)).astype(bool))
    payload = {k: v.pop(k) for k in ("sct", "fct2", "bct", "tbtp2") if k in v}
    return rp.profile.Message("test", "table", {**v, "payload": payload}, None, 0, True, 0, 0.0, None)

def _network_plan():
    plan = rp.plan.load("demo/mftdma")
    plan.sequences[0].start, plan.sequences[0].count = 10**9 + 27000, 100
    return plan

def test_tables_round_trip_to_the_same_plan(rng):
    plan = _network_plan()
    plan.assign(0, 101, 0, [(11, 0), (0, 0), (12, 0)], offset=3)
    b = rp.plan.builder("dvbrcs2/forward")
    for proto, tb in b.tables(plan, network_id=9, superframes=[(0, 101)]):
        assert proto == 0x0082
        b.feed(_table_message(tb))
    q = b.plan
    s = plan.sequences[0]
    key = lambda w: (w.superframe, w.frame, w.slot, w.freq, w.start, w.burst, w.tx_type, w.assignment)
    assert [key(w) for w in q.windows(s.start, s.start + 3*s.duration)] == [key(w) for w in plan.windows(s.start, s.start + 3*s.duration)]
    assert [(w.slot, w.assignment) for w in q.windows(s.start + s.duration, s.start + 2*s.duration) if w.frame == 0] == \
        [(3, 11), (4, 0), (5, 12)]
    assert q.tx_types[128].fields == {"bits": 320, "constellation": "psk/qpsk", "threshold": 0, "uw": 0xcf07d68afc0bd6f3}
    assert q.tx_types[129].btus == 5 and q.frame_types[1].btu_bw == 337500.0 and q.sequences[0].frames[0].offset == -1750000.0

def test_ncr_packets_pair_with_the_sof_two_frames_before():
    b = rp.plan.builder("dvbrcs2/forward")
    for k in range(6):
        b.feed(rp.profile.Message("t", "pls", {"modcod": 4}, None, 0, True, 1000*k + 7, 0.0, None))
        if k >= 2: # (the packet in frame k carries frame k-2's SOF)
            b.feed(rp.profile.Message("t", "gse", {"protocol": 0x0081}, b.reference(5e9 + 27*(1000*(k - 2) + 7))[1], 48, True, 1000*k + 300, 0.0, None))
    a, r = b.fit()
    assert r == pytest.approx(27) and b.clock(0) == pytest.approx(5e9)

def test_ncr_packet_wraps_base_and_extension():
    b = rp.plan.builder("dvbrcs2/forward")
    proto, pdu = b.reference(123456789*300 + 299)
    assert proto == 0x0081 and len(pdu) == 6 and b.value(pdu) == 123456789*300 + 299
    assert b.value(b.reference((1 << 33)*300 + 5)[1]) == 5 # (the base wraps)

def test_builder_and_plan_definitions_are_not_mixed_up():
    with pytest.raises(rp.errors.ConfigError, match="builder"):
        rp.plan.load("dvbrcs2/forward")
    with pytest.raises(rp.errors.ConfigError, match="not a builder"):
        rp.plan.builder("demo/mftdma")

# ---- the whole network

N0, DELAY = 10**9, 1e-4

def _network(rng, superframes=3, per=5, frames=6):
    plan = _network_plan()
    s = plan.sequences[0]
    items = []
    for k in range(superframes):
        ws = plan.windows(s.start + k*s.duration, s.start + (k + 1)*s.duration)
        assigned = {}
        for i in sorted(rng.choice(len(ws), per, replace=False)):
            w, term = ws[i], int(rng.integers(1, 250))
            items.append((w, {"superframe": w.superframe, "frame": w.frame, "slot": w.slot, "terminal": term, "seq": len(items)},
                          rng.integers(0, 2, plan.tx_types[w.tx_type].fields["bits"] - 32).astype(bool)))
            assigned.setdefault(w.frame, {})[w.slot] = term
        for fr, slots in assigned.items():
            lo, hi = min(slots), max(slots)
            plan.assign(0, ws[0].superframe, fr, [(slots.get(j, 0), 0) for j in range(lo, hi + 1)], offset=lo)
    # Forward: tables and NCR in GSE in DVB-S2 (QPSK 1/2 with pilots, 1 MBd at 4 samples), the NCR of frame k-2's SOF in frame k
    dfl = rp.profile.table("dvbs2/modcod").row({"modcod": 4, "short": 0, "pilots": 1})["kbch"] - 80
    hub = rp.plan.builder("dvbrcs2/forward")
    tabs = hub.tables(plan, network_id=7, superframes=[(0, s.count + k) for k in range(superframes)])
    fwd = rp.profile.load("dvbs2/normal")
    def send(sof):
        data = []
        for k in range(frames):
            pdus = ([hub.reference(N0 + 27e6*sof[k - 2]/4e6)] if k >= 2 else []) + tabs
            data.append(np.unpackbits(np.frombuffer(gse.pack(pdus, dfl//8)[0], np.uint8)).astype(bool))
        return fwd.tx()(rp.Data(np.concatenate(data), Fs=1e6, sps=1, flush=True), fields=[{"modcod": 4, "pilots": 1}]*frames)
    o = send([0]*frames)
    sof = [o.meta.delay*4e6 + k*(90 + 360*90 + 22*36)*4 for k in range(frames)]
    xf = np.asarray(send(sof).data)
    xf = (xf*np.exp(0.7j) + 0.05*(rng.normal(size=len(xf)) + 1j*rng.normal(size=len(xf)))).astype(np.complex64)
    # Return: the profile's own bursts on the plan, arriving DELAY late (a path the receiver is not told)
    tx = rp.profile.load("demo/mftdma").tx()
    tx.burst_tx.plan, tx.burst_tx.origin = plan, N0 - DELAY*27e6
    items.sort(key=lambda it: it[0].burst)
    xr = np.concatenate([np.asarray(tx(rp.Data(pay, sps=1), fields=f).data) for _, f, pay in items] + [np.asarray(tx.flush().data)])
    xr = np.concatenate((xr, np.zeros(4000)))
    xr = (xr + 0.02*(rng.normal(size=len(xr)) + 1j*rng.normal(size=len(xr)))).astype(np.complex64)
    return xf, xr, items

def test_the_forward_link_tells_the_return_link_its_plan(rng):
    xf, xr, items = _network(rng)
    fwd, b = rp.profile.load("dvbrcs2/forward").rx(), rp.plan.builder("dvbrcs2/forward")
    for blk in rp.source.Memory(xf, Fs=4e6, sps=4)(1 << 16):
        blk + fwd
        for m in fwd.take_messages():
            b.feed(m)
    assert b.plan is not None and b.fit()[1]*4e6 == pytest.approx(27e6, rel=1e-6)
    assert abs(b.clock(0) - N0) < 27e6*20e-6 # (the forward link's own filters: within 20 us)
    rx = rp.profile.load("demo/mftdma").rx()
    rx.burst.control(plan=b.plan, origin=b.clock(0)) # the plan and the clock, from the air
    msgs = []
    for blk in rp.source.Memory(xr, Fs=8e6)(1 << 15):
        blk + rx
        msgs += rx.take_messages()
    got = {(m.fields["terminal"], m.fields["seq"]): m.fields["burst"] for m in msgs if m.kind == "hdr" and m.crc_ok}
    for w, f, _ in items:
        bb = got[(f["terminal"], f["seq"])]
        assert (bb["superframe"], bb["frame"], bb["slot"], bb["tx_type"], bb["assignment"]) == (w.superframe, w.frame, w.slot, w.tx_type, f["terminal"])
