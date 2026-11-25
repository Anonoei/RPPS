"""A frame's parts (rpps.profile.message: Message.frame, parts, check): an HDLC frame's own header stripped by a link
header after it (profile demo/hdlc_header), 802.15.4 frames (ieee802154/frames), whatever the block cut"""
import struct

import numpy as np
import pytest

import rpps as rp

def run(p, x, Fs, sps, delay=0.0, cuts=None):
    rx, st, msgs, i = p.rx(), None, [], 0
    cuts = cuts or [len(x)]
    for k, n in enumerate(cuts):
        d = rp.Data(x[i:i + n], Fs=Fs, sps=sps, delay=delay, stream=st, flush=k == len(cuts) - 1)
        st, i = d.stream, i + n
        rx(d)
        msgs += rx.take_messages()
    return msgs

PAYLOADS = [b"hello world", b"second frame!", b"third"]

@pytest.fixture(scope="module")
def hdlc_header():
    p = rp.profile.load("demo/hdlc_header", centre=0, sps=10)
    tx, s, waves = p.tx(), None, []
    for i, f in enumerate(PAYLOADS):
        d = rp.Data(f, Fs=1200.0, sps=1, stream=s, flush=i == len(PAYLOADS) - 1)
        s = d.stream
        o = tx(d, fields={"address": 7, "control": 3, "protocol": 0x21 + i})
        waves.append(o.data)
    x = np.concatenate([np.zeros(400, np.complex64)] + waves + [np.zeros(400, np.complex64)])
    return p, x, o

@pytest.mark.parametrize("cuts", [None, [1000, 1, 2500, 7, 1501]])
def test_a_header_inside_an_hdlc_frame_is_stripped(hdlc_header, cuts):
    p, x, o = hdlc_header
    cuts = cuts if cuts is None else cuts + [len(x) - sum(cuts)]
    msgs = run(p, x, o.Fs, o.sps, o.delay, cuts)
    frames = [m for m in msgs if m.kind == "frame"]
    hdrs = [m for m in msgs if m.kind == "hdr"]
    assert [m.payload for m in hdrs] == PAYLOADS
    assert [m.fields for m in hdrs] == [{"address": 7, "control": 3, "protocol": 0x21 + i} for i in range(3)]
    assert [m.sample for m in hdrs] == [m.sample for m in frames] # Placed where their frames start
    for f, h in zip(frames, hdrs):
        assert f.check == "crc16_x25" and f.crc_ok and f.payload == h.frame # The frame's payload is the header's frame
        assert f.parts == {"payload": (0, 8*len(f.payload)), "fcs": (8*len(f.payload), 16)}
        assert f.frame[:len(f.payload)] == f.payload and len(f.frame) == len(f.payload) + 2
        assert h.parts == {"header": (0, 24), "payload": (24, 8*len(h.payload))}
        assert h.check is None and h.crc_ok is None # (the header has no check of its own)

def test_the_capture_rule_takes_header_and_payload(hdlc_header):
    from rpps.link.capture import unexported_pdu
    p, x, o = hdlc_header
    packets = p.capture.packets(run(p, x, o.Fs, o.sps, o.delay))
    assert [k.kind for k in packets] == ["hdr"]*3 # ("frame": false)
    assert [unexported_pdu(k.data)[1] for k in packets] == [bytes([7, 3, 0x21 + i]) + f for i, f in enumerate(PAYLOADS)]

def test_80215_4_frames():
    p = rp.profile.load("ieee802154/frames", sync={"timing": {"method": "known"}})
    crc = rp.link.load("crc/crc16_kermit").encoder
    macs = []
    for i in range(3):
        m = bytes.fromhex("41c8") + bytes([i]) + bytes.fromhex("3412ffff0100") + f"frame {i}".encode()
        macs.append(m + struct.pack("<H", crc.compute(m)))
    tx, s, waves = p.tx(), None, []
    for i, m in enumerate(macs):
        d = rp.Data(m, Fs=250e3, sps=1, stream=s, flush=i == len(macs) - 1)
        s = d.stream
        o = tx(d)
        waves.append(o.data)
    x = np.concatenate([np.zeros(256, np.complex64)] + waves + [np.zeros(256, np.complex64)]) # (on a symbol: see the profile)
    for cuts in (None, [3000, 7, 20000, len(x) - 23007]):
        msgs = run(p, x, o.Fs, o.sps, o.delay, cuts)
        assert [m.payload for m in msgs] == macs and [m.fields["frame_length"] for m in msgs] == [len(m) for m in macs]
    packets = p.capture.packets(msgs)
    assert [k.linktype.name for k in packets] == ["ieee802_15_4"]*3 and [k.data for k in packets] == macs
