"""Captures (rpps.link.capture): link types, transforms and their inverses, finding a link type in a frame, the shipped
profiles' rules checked against Wireshark's own dissectors when tshark is installed"""
import shutil
import struct
import subprocess
import zlib
import numpy as np
import pytest

import rpps as rp
from rpps.errors import ConfigError
from rpps.link.capture import Capture, Captures, linktype, find, exported_pdu, unexported_pdu
from rpps.profile.message import Message

TSHARK = shutil.which("tshark")
needs_tshark = pytest.mark.skipif(TSHARK is None, reason="tshark is not installed")

def tshark(path, *fields, opts=()):
    """Rows of tshark's fields for each packet of a capture"""
    args = [TSHARK, "-r", str(path), *opts, "-T", "fields"] + [x for f in fields for x in ("-e", f)]
    out = subprocess.run(args, capture_output=True, text=True, check=True).stdout
    return [line.split("\t") for line in out.splitlines()]

def message(frame, parts, fields=None, ok=True, kind="frame", payload=None):
    m = Message("demo/x", kind, fields or {}, payload, 0, ok, 100, 0.5)
    m.frame, m.frame_bits, m.parts = frame, 8*len(frame), parts
    return m

# ---- link types

def test_link_types_load():
    names = rp.link.capture.linktypes()
    assert {"ethernet", "raw", "ieee802_15_4", "ble_ll_phdr", "ax25", "mpeg2_ts", "upper_pdu", "user0"} <= set(names)
    lt = linktype("ieee802_15_4_nofcs")
    assert (lt.dlt, lt.arphrd, lt.proto) == (230, 804, 0xF6)
    assert linktype("ethernet").dlt == 1 and linktype("upper_pdu").dlt == 252

def ipv4(rng, n=None):
    """A valid IPv4 packet (header checksum and length right) with a random payload"""
    n = int(rng.integers(0, 200)) if n is None else n
    h = bytearray(struct.pack(">BBHHHBBH4s4s", 0x45, 0, 20 + n, int(rng.integers(0, 65536)), 0x4000, 64, 17, 0,
                              rng.bytes(4), rng.bytes(4)))
    h[10:12] = struct.pack(">H", rp.link.load("crc/internet16").encoder.compute(bytes(h)))
    return bytes(h) + rng.bytes(n)

def test_signatures_score():
    rng = np.random.default_rng(3)
    ip = ipv4(rng, 30)
    assert linktype("ipv4").score(ip) == 2 # Fields valid, header checksum holds
    bad = bytearray(ip)
    bad[12] ^= 1
    assert linktype("ipv4").score(bytes(bad)) == 1 # (the checksum doesn't)
    assert linktype("ipv4").score(ip + b"x") == 0 # Its length isn't what is left
    eth = rng.bytes(12) + b"\x08\x00" + ip
    assert linktype("ethernet").score(eth) == 3 # One more than the IPv4 inside
    assert linktype("ethernet").score(rng.bytes(12) + b"\x88\xb5" + rng.bytes(30)) == 1

def test_find_after_an_unknown_header():
    rng = np.random.default_rng(4)
    ip = ipv4(rng, 40)
    eth = rng.bytes(12) + b"\x08\x00" + ip
    head = b"\x45\x00" + rng.bytes(9) # (starts like IPv4: the checksum and length tell it isn't)
    off, lt = find(head + eth, ["ethernet", "ipv4", "ipv6"])
    assert (off, lt.name) == (len(head), "ethernet") # The Ethernet frame, not the IPv4 packet inside it
    off, lt = find(head + ip, ["ethernet", "ipv4", "ipv6"])
    assert (off, lt.name) == (len(head), "ipv4")
    assert find(head + ip, ["ethernet"], least=2) == (None, None)

def test_find_rates():
    """Measured over 500 frames each (rates in the docstring of rpps.link.capture): every frame found behind a random
    header of 0..64 bytes; random bytes match at score 2 never (asserted below 0.5%), at score 1 about 0.8% (below 3%)"""
    rng = np.random.default_rng(0)
    hits = 0
    for i in range(300):
        k, pkt = int(rng.integers(0, 65)), ipv4(rng)
        kind = "ipv4" if i % 2 else "ethernet"
        if kind == "ethernet":
            pkt = rng.bytes(12) + b"\x08\x00" + pkt
        off, lt = find(rng.bytes(k) + pkt, ["ethernet", "ipv4", "ipv6"])
        hits += off == k and lt is not None and lt.name == kind
    assert hits == 300
    junk = [rng.bytes(int(rng.integers(40, 400))) for _ in range(400)]
    assert sum(find(j, ["ethernet", "ipv4", "ipv6"], least=2)[1] is not None for j in junk) <= 2
    assert sum(find(j, ["ethernet", "ipv4", "ipv6"], least=1)[1] is not None for j in junk) <= 12

# ---- rules and transforms

def test_parts_are_included_in_order():
    m = message(b"HDRpayloadFC", {"header": (0, 24), "payload": (24, 56), "fcs": (80, 16)})
    assert Capture({"linktype": "raw", "include": ["payload"]}).packet(m).data == b"payload"
    assert Capture({"linktype": "raw", "include": ["header", "payload", "fcs"]}).packet(m).data == b"HDRpayloadFC"
    assert Capture({"linktype": "raw", "include": ["payload", "fcs"]}).packet(m).data == b"payloadFC"
    # A message without the whole frame: its payload is the frame
    plain = Message("demo/x", "frame", {}, b"abc", 24, True)
    assert Capture({"linktype": "raw"}).packet(plain).data == b"abc"

@pytest.mark.parametrize("transform,expect", [
    ({"bit_order": "lsb"}, bytes([0x80, 0x01]) + b""),
    ({"prefix": "0x8E89BED6", "bytes": 4, "order": "little"}, bytes.fromhex("d6be898e") + b"\x01\x80"),
    ({"suffix": "addr", "bytes": 2}, b"\x01\x80\x12\x34"),
    ({"strip": {"head": 1}}, b"\x80"),
    ({"set": {"at": 0, "value": "0x47"}, "was": "0xB8"}, b"\x47\x80"),
    ({"fcs": "crc16_kermit", "order": "little"}, b"\x01\x80" + struct.pack("<H", rp.link.load("crc/crc16_kermit").encoder.compute(b"\x01\x80"))),
])
def test_transforms_and_their_inverses(transform, expect):
    m = message(b"\x01\x80", {"payload": (0, 16)}, fields={"addr": 0x1234})
    c = Capture({"linktype": "raw", "transform": [transform]}, params={})
    p = c.packet(m)
    assert p.data == expect
    back, fields = c.unpacket(p.data)
    if "strip" in transform:
        assert back == b"\x00\x80" # (what was stripped comes back as the fill)
    elif "set" in transform:
        assert back == b"\xb8\x80"
    else:
        assert back == b"\x01\x80"
    if "suffix" in transform:
        assert fields == {"addr": 0x1234}

def test_a_bad_fcs_is_refused_on_transmit():
    c = Capture({"linktype": "raw", "transform": [{"fcs": "crc16_kermit"}]})
    with pytest.raises(ValueError, match="FCS"):
        c.unpacket(b"\x01\x80\x00\x00")

def test_ppp_restore():
    c = Capture({"linktype": "ppp", "transform": [{"restore": "ppp_acfc_pfc"}]})
    assert c.packet(message(b"\x21hi", {"payload": (0, 24)})).data == b"\xff\x03\x00\x21hi" # ACFC and PFC
    assert c.packet(message(b"\xff\x03\x00\x21hi", {"payload": (0, 48)})).data == b"\xff\x03\x00\x21hi" # (neither)

def test_drop_and_fields():
    c = Capture({"linktype": "raw", "drop": "vcid == 63", "fields": ["vcid"]})
    assert c.packet(message(b"x", {"payload": (0, 8)}, {"vcid": 63})) is None
    p = c.packet(message(b"x", {"payload": (0, 8)}, {"vcid": 1, "other": 2}))
    assert p.fields == {"vcid": 1} and "vcid=1" in p.comment()

def test_find_rule():
    rng = np.random.default_rng(5)
    ip = ipv4(rng, 12)
    c = Capture({"linktype": "find", "find": {"in": ["ethernet", "ipv4"], "max": 16}})
    p = c.packet(message(b"\x01\x02\x03" + ip, {"payload": (0, 8*(3 + len(ip)))}))
    assert p.linktype.name == "ipv4" and p.data == ip
    p = c.packet(message(rng.bytes(40), {"payload": (0, 320)}))
    assert p.linktype.name == "upper_pdu" and c.undetected == 1 # Not found: whole, as exported PDU data

def test_exported_pdu_round_trip():
    raw = exported_pdu(b"\x01\x02", "lte_rrc.bcch_bch", "lte/downlink pbch", "ok sfn=3")
    assert unexported_pdu(raw) == ("lte_rrc.bcch_bch", b"\x01\x02")
    assert len(raw) % 4 == 2 # (tags padded to 4 bytes; the payload follows as it is)

def test_rule_errors():
    with pytest.raises(ConfigError, match="unknown keys"):
        Capture({"linktype": "raw", "nope": 1})
    with pytest.raises(ConfigError, match="include"):
        Capture({"linktype": "raw", "include": ["body"]})
    with pytest.raises(ConfigError, match="pseudo-header"):
        Capture({"linktype": "raw", "transform": [{"header": "nope"}]})
    with pytest.raises(ConfigError, match="find"):
        Capture({"linktype": "find"})
    with pytest.raises(ConfigError, match="kinds it doesn't give"):
        p = rp.profile.load("ax25/afsk1200")
        p._captures({"nope": {"linktype": "ax25"}})

def test_kinds_without_a_rule_are_exported_pdu_data():
    caps = Captures({"frame": False})
    assert caps.packet(message(b"x", {"payload": (0, 8)})) is None
    p = caps.packet(message(b"x", {"payload": (0, 8)}, kind="other"))
    assert p.linktype.name == "upper_pdu" and unexported_pdu(p.data) == ("data", b"x")

# ---- the shipped profiles, as Wireshark sees them

def ble_message(p, adv, fields):
    o = p.tx()([rp.Data(np.unpackbits(np.frombuffer(adv, np.uint8)).astype(bool), Fs=1e6, sps=1)], fields=fields,
               flush=True)
    rx = p.rx()
    rx(rp.Data(np.asarray(o.data), Fs=o.Fs, sps=4, flush=True))
    return rx.take_messages()

ADV = bytes.fromhex("112233445566") + bytes.fromhex("020106") + bytes([5, 9]) + b"RPPS"

def test_ble_capture_bytes():
    p = rp.profile.load("ble/adv")
    (m,) = ble_message(p, ADV, {"type": 2, "txadd": 1})
    assert m.check == "crc24_ble" and m.parts == {"header": (0, 16), "payload": (16, 120), "fcs": (136, 24)}
    pk = p.capture.packet(m)
    assert pk.linktype.name == "ble_ll_phdr"
    assert pk.data[:10] == bytes([0, 0, 0, 0]) + bytes(4) + (0x0C01).to_bytes(2, "little") # RF channel 0 (index 37), CRC ok
    assert pk.data[10:14] == bytes.fromhex("d6be898e") and pk.data[14:16] == b"\x42\x0f" and pk.data[16:31] == ADV

@needs_tshark
def test_wireshark_reads_the_captures(tmp_path):
    """BLE (its CRC checked by Wireshark), AX.25, 802.11b with radiotap (FCS present), as each profile captures them"""
    out = []
    # BLE, plain link layer: Wireshark checks the CRC itself
    b = rp.profile.load("ble/adv")
    (m,) = ble_message(b, ADV, {"type": 2, "txadd": 1})
    plain = Capture({"linktype": "ble_ll", "include": ["header", "payload", "fcs"],
                     "transform": [{"prefix": "0x8E89BED6", "bytes": 4, "order": "little"}]})
    out += [plain.packet(m), b.capture.packet(m)]
    # AX.25 UI frame
    call = lambda c, ssid, last: bytes(ord(ch) << 1 for ch in c.ljust(6)) + bytes([0x60 | ssid << 1 | last])
    frame = call("APRS", 0, 0) + call("N0CALL", 7, 1) + b"\x03\xf0" + b"!4903.50N/07201.75W-"
    a = rp.profile.load("ax25/afsk1200", centre=0, sps=10)
    o = a.tx()(rp.Data(frame, Fs=1200.0, sps=1, flush=True))
    rx = a.rx()
    rx(rp.Data(np.concatenate((np.zeros(400, np.complex64), o.data, np.zeros(400, np.complex64))), Fs=o.Fs, sps=o.sps,
               delay=o.delay, flush=True))
    out += a.capture.packets(rx.take_messages())
    # 802.11b data frame with its FCS
    w = rp.profile.load("ieee80211b/long", sync={"timing": {"method": "known"}, "carrier": {"method": "none"}})
    mpdu = bytes.fromhex("0802") + bytes(2) + bytes.fromhex("ffffffffffff020000000001020000000002") + bytes(2) + b"hello"
    mpdu += struct.pack("<I", zlib.crc32(mpdu))
    o = w.tx()(rp.Data(mpdu, Fs=1e6, sps=1, flush=True), fields={"signal": 20})
    rx = w.rx()
    rx(rp.Data(np.concatenate((np.zeros(100, np.complex64), o.data, np.zeros(400, np.complex64))), Fs=o.Fs, sps=o.sps,
               delay=o.delay, flush=True))
    out += w.capture.packets(rx.take_messages())
    with rp.source.PcapSink(tmp_path/"c.pcapng") as s:
        out >> s
    rows = tshark(tmp_path/"c.pcapng", "frame.protocols", "btle.crc.incorrect", "btle.advertising_address", "ax25.dst",
                  "ax25.src", "radiotap.datarate", "radiotap.flags.fcs", "wlan.fcs.status",
                  opts=("-o", "wlan.check_fcs:TRUE", "-o", "wlan.check_checksum:TRUE"))
    assert "btle" in rows[0][0] and rows[0][1] == "" and rows[0][2] == "66:55:44:33:22:11" # CRC correct
    assert "btle_rf:btle" in rows[1][0] and rows[1][2] == "66:55:44:33:22:11"
    assert rows[2][0].startswith("ax25") and rows[2][3:5] == ["APRS", "N0CALL-7"]
    assert rows[3][0].startswith("radiotap:wlan_radio:wlan") and rows[3][5] == "2" and rows[3][6] == "True"
    assert rows[3][7] == "1" # FCS good
