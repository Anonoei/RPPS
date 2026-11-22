"""Decoded messages (rpps.profile.message): each frame a profile receives, with its fields, payload, check and place"""
import json

import numpy as np
import pytest

import rpps as rp

def run(p, x, Fs, sps, delay=0.0, cuts=None, cf=0.0):
    """The messages of a capture through a profile's receiver, cut into blocks"""
    rx, st, msgs, i = p.rx(), None, [], 0
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

def runner_messages(name, x, Fs, block, **params):
    """Messages of a branching profile run block by block on a capture (rp.source.Memory)"""
    rx, msgs = rp.profile.load(name, **params).rx(), []
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
