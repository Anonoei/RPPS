"""Decoded messages (rpps.profile.message): each frame a profile receives, with its fields, payload, check and place"""
import json

import numpy as np
import pytest

import rpps as rp

def run(p, x, Fs, sps, delay=0.0, cuts=None, cf=0.0, kinds=None):
    """The messages of a capture through a profile's receiver, cut into blocks"""
    rx, st, msgs, i = p.rx(kinds=kinds), None, [], 0
    cuts = cuts or [len(x)]
    for k, n in enumerate(cuts):
        d = rp.Data(x[i:i + n], Fs=Fs, sps=sps, delay=delay, cf=cf, stream=st, flush=k == len(cuts) - 1)
        st, i = d.stream, i + n
        rx(d)
        msgs += rx.take_messages()
    return msgs

@pytest.fixture(scope="module")
def adsb():
    rng = np.random.default_rng(12)
    p = rp.profile.load("adsb/mode_s")
    payloads = [rng.integers(0, 2, 83).astype(bool) for _ in range(3)]
    tx, s, waves, starts, at = p.tx(), None, [], [], 25
    for i, m in enumerate(payloads):
        d = rp.Data(m, Fs=2e6, sps=1, stream=s, flush=i == len(payloads) - 1)
        s = d.stream
        o = tx(d, fields={"df": 17})
        starts.append(at)
        noise = (0.02*(rng.normal(size=301) + 1j*rng.normal(size=301))).astype(np.complex64)
        waves += [o.data*(0.3 + i*0.5), noise]
        at += len(o.data) + len(noise)
    return p, np.concatenate([np.zeros(25, np.complex64)] + waves), o, payloads, starts

def test_each_adsb_frame_is_a_message(adsb):
    p, x, o, payloads, starts = adsb
    msgs = run(p, x, 2e6, 1, o.delay, cf=1090e6)
    assert [m.kind for m in msgs] == ["frame"]*3 and all(m.crc_ok for m in msgs)
    for m, bits, start in zip(msgs, payloads, starts):
        assert m.profile == "adsb/mode_s" and m.fields["df"] == 17 and m.bits == 83
        assert m.payload == np.packbits(bits).tobytes()
        assert 0 <= m.sample - start <= 16 + 5*2 # From the frame's header: after the 16-chip preamble, give or take
        assert m.time == pytest.approx(m.sample/2e6) and m.freq == 1090e6
    d = json.loads(json.dumps(msgs[0].to_dict()))
    assert d["payload"] == msgs[0].payload.hex() and d["crc_ok"] is True and d["fields"]["df"] == 17

def test_messages_do_not_depend_on_the_block_cut(adsb):
    p, x, o, payloads, starts = adsb
    whole = run(p, x, 2e6, 1, o.delay)
    cut = run(p, x, 2e6, 1, o.delay, cuts=[100, 7, 500, 33, len(x) - 640])
    assert [(m.sample, m.payload, m.crc_ok) for m in cut] == [(m.sample, m.payload, m.crc_ok) for m in whole]

def test_a_damaged_frame_is_a_failed_message(adsb):
    p, x, o, payloads, starts = adsb
    y = x.copy()
    k = 25 + 16 + 40 # Two halves of one bit of the first frame's data swapped: that bit inverted, the parity fails
    y[k], y[k + 1] = y[k + 1], y[k]
    msgs = run(p, y, 2e6, 1, o.delay)
    assert [m.crc_ok for m in msgs] == [False, True, True] and msgs[0].fields["df"] == 17

@pytest.fixture(scope="module")
def ble():
    rng = np.random.default_rng(11)
    p = rp.profile.load("ble/adv")
    pdus = [rng.integers(0, 2, 8*n).astype(bool) for n in (6, 20, 37)]
    tx, s, waves, starts, at = p.tx(), None, [], [], 36
    for pdu in pdus:
        d = rp.Data(pdu, Fs=1e6, sps=1, stream=s, flush=True)
        s = d.stream
        o = tx(d, fields={"type": 2, "txadd": 1})
        starts.append(at)
        waves += [o.data, np.zeros(4000, np.complex64)]
        at += len(o.data) + 4000
    return p, np.concatenate([np.zeros(36, np.complex64)] + waves), pdus, starts

def test_ble_packets_are_messages_placed_in_the_capture(ble):
    """A burst profile: each burst runs in a stream of its own; its messages are placed back in the capture's"""
    p, x, pdus, starts = ble
    msgs = run(p, x, 4e6, 4)
    assert [m.kind for m in msgs] == ["pdu"]*3 and all(m.crc_ok for m in msgs)
    for m, pdu, start in zip(msgs, pdus, starts):
        assert m.fields["type"] == 2 and m.fields["txadd"] == 1 and m.payload == np.packbits(pdu).tobytes()
        assert 0 <= m.sample - start < 4*(8 + 32 + 16 + 8) # After the preamble and access address, at 4 samples a bit

def runner_messages(name, x, Fs, block, kinds=None, **params):
    """Messages of a branching profile run block by block on a capture (rp.source.Memory)"""
    rx, msgs = rp.profile.load(name, **params).rx(kinds=kinds), []
    for d in rp.source.Memory(x, Fs=Fs)(block):
        rx(d)
        msgs += rx.take_messages()
    return msgs

def test_lte_mib_cfi_dci_and_sib1_are_messages():
    from rpps.grid import lte
    sib = np.random.default_rng(2).integers(0, 2, 224).astype(np.uint8)
    x = lte.tx.channel(lte.tx.downlink(pci=10, ports=1, n_rb=6, frames=4, sib=sib), cfo=1.2e3, fs=1.92e6, delay=700,
                       snr_db=20, seed=3)
    msgs = runner_messages("lte/downlink", x, 1.92e6, 20000, n_rb=6)
    by = {k: [m for m in msgs if m.kind == k] for k in ("pbch", "pcfich", "pdcch", "pdsch")}
    assert all(by.values()) and {m.kind for m in msgs} <= set(by)
    mib = [m for m in by["pbch"] if m.crc_ok]
    assert mib and all(m.fields["pci"] == 10 and m.fields["n_rb"] == 6 and m.fields["ports"] == 1 for m in mib)
    assert [m.fields["sfn"] for m in mib] == sorted(m.fields["sfn"] for m in mib)
    assert all(m.crc_ok is None and m.fields["cfi"] in (1, 2, 3) for m in by["pcfich"])
    assert all(m.fields["rnti"] == 0xFFFF and m.fields["format"] in ("1A", "1C") for m in by["pdcch"])
    sib1 = [m for m in by["pdsch"] if m.crc_ok]
    assert sib1 and all(m.payload == np.packbits(sib).tobytes() and m.bits == 224 for m in sib1)
    assert [m.sample for m in msgs] == sorted(m.sample for m in msgs) and 0 <= msgs[0].sample and msgs[-1].sample < len(x)

def test_nr_mib_dci_and_sib1_are_messages():
    from rpps.grid import nr
    rng = np.random.default_rng(2)
    sib = rng.integers(0, 2, nr.tx.sib_size(24, 1, 2, 2)).astype(np.uint8)
    y = nr.tx.downlink(pci=17, nfft=512, frames=10, sib=sib, ssb=(0, 1), mcs=2, k_ssb=2)
    x = nr.tx.awgn(np.concatenate((np.zeros(3000), y)), 15, ref=y)
    msgs = runner_messages("nr/downlink", x, 7.68e6, 100000)
    mib = [m for m in msgs if m.kind == "pbch" and m.crc_ok]
    sib1 = [m for m in msgs if m.kind == "pdsch" and m.crc_ok]
    assert len(mib) == 5 and all(m.fields["pci"] == 17 and m.fields["k_ssb"] == 2 for m in mib)
    assert len(sib1) == 5 and all(m.payload == np.packbits(sib).tobytes() for m in sib1)
    assert all(m.fields["rnti"] == 0xFFFF for m in msgs if m.kind == "pdcch")

KNOWN = {"timing": {"method": "known"}, "carrier": {"method": "none"}}

def test_ax25_frames_are_messages_with_their_fcs():
    rng = np.random.default_rng(13)
    p = rp.profile.load("ax25/afsk1200", centre=0, sps=10)
    pk = [bytes(rng.integers(0, 256, n).astype(np.uint8)) for n in (20, 45, 30)]
    tx, s, waves = p.tx(), None, []
    for i, f in enumerate(pk):
        d = rp.Data(f, Fs=1200.0, sps=1, stream=s, flush=i == len(pk) - 1)
        s = d.stream
        o = tx(d)
        waves.append(o.data)
    x = np.concatenate([np.zeros(400, np.complex64)] + waves + [np.zeros(400, np.complex64)])
    x = (x + 0.05*(rng.normal(size=len(x)) + 1j*rng.normal(size=len(x)))).astype(np.complex64)
    for cuts in (None, [1000, 1, 2500, len(x) - 3501]):
        msgs = run(p, x, o.Fs, o.sps, o.delay, cuts=cuts)
        assert [m.payload for m in msgs] == pk and all(m.crc_ok and m.kind == "frame" for m in msgs)
        assert [m.fields["bytes"] for m in msgs] == [20, 45, 30]
        assert msgs[0].sample >= 400 and [m.sample for m in msgs] == sorted(m.sample for m in msgs)

def test_ccsds_transfer_frames_are_messages():
    rng = np.random.default_rng(14)
    p = rp.profile.load("ccsds/tm_rs", sync=KNOWN, depth=2)
    frames = rng.integers(0, 2, 3*8*223*2).astype(bool)
    o = p.tx()(rp.Data(frames, Fs=1e6, sps=1, flush=True))
    x = np.concatenate((np.zeros(37, np.complex64), -o.data, np.zeros(100, np.complex64)))
    msgs = run(p, x, o.Fs, 4, o.delay, cuts=[4000, 3, 9000, len(x) - 13003])
    per = 8*223*2
    assert [m.kind for m in msgs] == ["frame"]*3 and all(m.crc_ok for m in msgs)
    assert [m.payload for m in msgs] == [np.packbits(frames[i*per:(i + 1)*per]).tobytes() for i in range(3)]
    span = 4*(32 + 8*255*2) # Samples a frame: its ASM and its codewords at 4 samples a bit
    assert all(abs((m.sample - msgs[0].sample) - i*span) <= 4 for i, m in enumerate(msgs))

def test_dvbs_transport_packets_are_messages_with_their_pid():
    rng = np.random.default_rng(5)
    p = rp.profile.load("dvbs/qpsk_3_4", sync=KNOWN)
    pk = rng.integers(0, 256, (8*12, 188)).astype(np.uint8)
    pk[:, 0] = 0x47
    wave = p.tx()(rp.Data(pk.reshape(-1), Fs=1e6, sps=1, flush=True))
    msgs = run(p, wave.data, wave.Fs, wave.sps, wave.delay)
    assert len(msgs) >= len(pk) - 16 and all(m.kind == "packet" and m.bits == 1504 for m in msgs)
    k = next(k for k in range(len(pk)) if msgs[0].payload == pk[k].tobytes())
    for m, row in zip(msgs, pk[k:]):
        assert m.payload == row.tobytes() and m.crc_ok
        assert m.fields["pid"] == (int(row[1]) & 0x1F) << 8 | int(row[2]) and m.fields["tei"] == int(row[1]) >> 7

def test_a_transfer_frame_past_correction_is_a_failed_message():
    from rpps.base.types import Rep
    rng = np.random.default_rng(15)
    p = rp.profile.load("ccsds/tm_rs", sync=KNOWN, depth=2)
    frames = rng.integers(0, 2, 3*8*223*2).astype(bool)
    d = rp.Data(frames, Fs=1e6, sps=1, flush=True)
    for op in [op for how, op in p.tx() if how == "*"][:-1]: # Down to channel bits
        d = d*op
    bits = np.asarray(d.data).copy()
    per = 32 + 8*255*2
    for k in rng.choice(np.arange(per + 32, 2*per - 16, 16), 60, replace=False): # 60 bad bytes in the second frame
        bits[k:k + 16] ^= True
    rx = p.rx(start=Rep.CHANNEL)
    rx(rp.Data(bits, flush=True).derive(bits, rp.Category.CODE, rp.dtype.BITS, rep=Rep.CHANNEL))
    assert [m.crc_ok for m in rx.take_messages()] == [True, False, True]

S2_FRAMES = [{"modcod": 4, "pilots": 1}, {"modcod": 12, "pilots": 1}, {"modcod": 1, "pilots": 0}, {"modcod": 4, "pilots": 1}]

def test_dvbs2_bbframes_serial_and_parallel():
    """Serial, the BB header emits each frame with its fields; parallel, the region's kernels decode the header inside, so
    each frame result is the message: the same payloads, near the same places"""
    p = rp.profile.load("dvbs2/normal", sync={"timing": {"method": "known"}})
    t = rp.profile.table("dvbs2/modcod")
    dfl = [t.row({**f, "short": 0})["kbch"] - 80 for f in S2_FRAMES]
    data = np.random.default_rng(4).integers(0, 2, sum(dfl)).astype(bool)
    o = p.tx()(rp.Data(data, Fs=1e6, sps=1, flush=True), fields=S2_FRAMES)
    x = (np.concatenate((o.data, np.zeros(200, np.complex64)))*np.exp(0.7j)).astype(np.complex64)
    msgs = run(p, x, o.Fs, 4, o.delay)
    pl = [m for m in msgs if m.kind == "pls"] # (the PL header step's id)
    assert [m.fields["modcod"] for m in pl[:4]] == [f["modcod"] for f in S2_FRAMES] and all(m.crc_ok for m in pl[:4])
    serial = [m for m in msgs if m.kind == "bbheader"]
    assert len(serial) == 4 and all(m.crc_ok for m in serial)
    cuts = np.cumsum([0] + dfl)
    assert [m.payload for m in serial] == [np.packbits(data[a:b]).tobytes() for a, b in zip(cuts[:-1], cuts[1:])]
    assert [m.bits for m in serial] == dfl
    rx = p.rx(parallel=rp.parallel.Parallel("inline"))
    rx(rp.Data(x, Fs=o.Fs, sps=4, delay=o.delay, flush=True))
    par = rx.take_messages()
    rx.close()
    par = [m for m in par if m.kind == "frame"] # (the PL headers, before the region, are messages too)
    assert len(par) == 4 and all(m.crc_ok for m in par)
    assert [m.payload for m in par] == [m.payload for m in serial]
    assert all(abs(a.sample - b.sample) <= 4*90 for a, b in zip(par, serial)) # (within a PL header)


# ---- per-kind filtering: only the kinds asked for, and only the steps they need

def same(msgs):
    return [(m.kind, m.sample, m.fields, m.payload, m.crc_ok) for m in msgs]

def test_the_kinds_each_profile_gives():
    kinds = lambda name, **kw: sorted(rp.profile.load(name).kinds(**kw))
    assert kinds("adsb/mode_s") == ["frame"]
    assert kinds("dvbs/qpsk") == ["packet"]
    assert kinds("ccsds/tm_rs") == ["frame"]
    assert kinds("lte/downlink") == ["pbch", "pcfich", "pdcch", "pdsch"]
    assert kinds("dvbs2/normal") == ["bbheader", "pls"]
    assert kinds("dvbs2/normal", parallel=True) == ["frame", "pls"] # (the region's kernels give one message a frame)

def test_a_kind_needs_its_step_and_the_headers_that_drive_it():
    p = rp.profile.load("lte/downlink")
    assert p.needed({"pbch"}) == {"pbch"}
    assert p.needed({"pdcch"}) == {"pbch", "pcfich", "pdcch"}
    assert p.needed({"pdsch"}) == {"pbch", "pcfich", "pdcch", "pdsch"}
    rx = p.rx(kinds={"pbch"})
    assert list(rx.chains) == ["pbch"]
    with pytest.raises(rp.errors.ConfigError, match="no message kinds"):
        p.rx(kinds={"mib"})

def test_lte_kinds_decode_as_they_do_with_every_step():
    from rpps.grid import lte
    sib = np.random.default_rng(2).integers(0, 2, 224).astype(np.uint8)
    x = lte.tx.channel(lte.tx.downlink(pci=10, ports=1, n_rb=6, frames=4, sib=sib), cfo=1.2e3, fs=1.92e6, delay=700,
                       snr_db=20, seed=3)
    full = runner_messages("lte/downlink", x, 1.92e6, 20000, n_rb=6)
    for kinds in ({"pbch"}, {"pdcch"}, {"pdsch"}):
        got = runner_messages("lte/downlink", x, 1.92e6, 20000, kinds=kinds, n_rb=6)
        assert got and same(got) == same(m for m in full if m.kind in kinds)

def test_dvbs2_pl_headers_alone_skip_the_payload():
    p = rp.profile.load("dvbs2/normal", sync={"timing": {"method": "known"}})
    t = rp.profile.table("dvbs2/modcod")
    dfl = [t.row({**f, "short": 0})["kbch"] - 80 for f in S2_FRAMES]
    data = np.random.default_rng(4).integers(0, 2, sum(dfl)).astype(bool)
    o = p.tx()(rp.Data(data, Fs=1e6, sps=1, flush=True), fields=S2_FRAMES)
    x = (np.concatenate((o.data, np.zeros(200, np.complex64)))*np.exp(0.7j)).astype(np.complex64)
    rx = p.rx(kinds={"pls"})
    assert [s.step.id for s in rx.slots][-1] == "pls" # (no descrambling, demapping, LDPC, BCH)
    full = run(p, x, o.Fs, 4, o.delay)
    got = run(p, x, o.Fs, 4, o.delay, kinds={"pls"})
    assert len(got) >= 4 and same(got) == same(m for m in full if m.kind == "pls")
    par = p.rx(kinds={"pls"}, parallel=rp.parallel.Parallel("inline")) # (none of the region needed: not run)
    assert type(par) is rp.profile.runner.Runner

def test_one_stream_of_several():
    p = rp.profile.load("dvbs2/mis")
    rx = p.rx(kinds={"bb_b"})
    assert list(rx.chains) == ["b"]

def test_a_burst_profile_keeps_its_kinds(ble):
    p, x, pdus, starts = ble
    assert same(run(p, x, 4e6, 4, kinds={"pdu"})) == same(run(p, x, 4e6, 4))
