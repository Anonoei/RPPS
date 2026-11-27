"""Field schemas (rpps.link.fields): signed, scaled, enum, per-field order, when, bytes, text, groups, repeats; and where
they are used: a link header read as it comes, payload fields in messages (AX.25, BLE advertising, MPEG-TS)"""
import numpy as np
import pytest

import rpps as rp
from rpps import _config
from rpps.errors import ConfigError
from rpps.link.fields import Schema, Short
from support.signals import hard, as_data_bits

def bits_of(*pairs):
    """Bits of (value, width) pairs, most significant first"""
    return np.concatenate([[(v >> (n - 1 - i)) & 1 for i in range(n)] for v, n in pairs]).astype(bool)

def test_values_signed_scaled_enum_order():
    s = Schema([{"name": "a", "bits": 4, "signed": True}, {"name": "b", "bits": 8, "scale": 25, "offset": -1000},
                {"name": "t", "bits": 4, "order": "lsb", "enum": {"2": "two"}}])
    v, used = s.parse(bits_of((0b1110, 4), (48, 8), (0b0100, 4)))
    assert v == {"a": -2, "b": 200, "t": 2, "t.name": "two"} and used == 16
    assert np.array_equal(s.build({"a": -2, "b": 200, "t": 2}), bits_of((0b1110, 4), (48, 8), (0b0100, 4)))
    assert s.fixed_bits == 16

def test_when_bytes_text_and_expression_sizes():
    s = Schema([{"name": "n", "bits": 8}, {"name": "flag", "bits": 1}, {"name": "x", "bits": 7, "if": "flag == 1"},
                {"name": "data", "type": "bytes", "bytes": "n - 1"},
                {"name": "call", "type": "text", "bytes": 3, "shift": 1}])
    b = np.concatenate((bits_of((3, 8), (0, 1)), np.unpackbits(np.frombuffer(b"\xab\xcd" + bytes(c << 1 for c in b"K7 "),
                                                                             np.uint8)).astype(bool)))
    v, used = s.parse(b)
    assert v == {"n": 3, "flag": 0, "data": b"\xab\xcd", "call": "K7"} and used == len(b)
    assert s.fixed_bits is None
    with pytest.raises(Short):
        s.parse(b[:20])
    assert np.array_equal(s.build(v), b)

def test_groups_repeat_until_and_star():
    s = Schema([{"name": "count", "bits": 4},
                {"name": "items", "repeat": "count", "fields": [{"name": "v", "bits": 4}]},
                {"name": "addr", "until": "last == 1", "fields": [{"name": "a", "bits": 3}, {"name": "last", "bits": 1}]},
                {"name": "rest", "repeat": "*", "fields": [{"name": "r", "bits": 2}]}])
    b = bits_of((2, 4), (5, 4), (9, 4), (0b0100, 4), (0b1111, 4), (0b1001, 4))
    v, used = s.parse(b)
    assert v["items"] == [{"v": 5}, {"v": 9}] and v["addr"] == [{"a": 2, "last": 0}, {"a": 7, "last": 1}]
    assert v["rest"] == [{"r": 2}, {"r": 1}] and used == len(b)
    assert np.array_equal(s.build(v), b)

def test_from_positions_and_short_circuit():
    s = Schema([{"name": "afc", "from": 2, "bits": 2},
                {"name": "ad", "from": 4, "if": "(afc & 2) == 2", "fields": [
                    {"name": "length", "bits": 4}, {"name": "flags", "bits": 4, "if": "length > 0"},
                    {"name": "pcr", "bits": 4, "if": "length > 0 and (flags & 1) == 1"}]}])
    assert s.parse(bits_of((0b0010, 4), (0, 4)))[0] == {"afc": 2, "ad": {"length": 0}} # (no flags: `and` stops)
    assert s.parse(bits_of((0b0011, 4), (1, 4), (1, 4), (6, 4)))[0] == {"afc": 3, "ad": {"length": 1, "flags": 1, "pcr": 6}}
    assert s.parse(bits_of((0b0001, 4)))[0] == {"afc": 1}

def test_schema_errors():
    with pytest.raises(ConfigError, match="needs 'bits'"):
        Schema([{"name": "x"}])
    with pytest.raises(ConfigError, match="takes"):
        Schema([{"name": "x", "bits": 1, "nonsense": 2}])
    with pytest.raises(ConfigError, match="needs 'bytes'"):
        Schema([{"name": "x", "type": "bytes"}])

def test_a_header_sized_by_its_fields_any_block_cut(rng):
    """A link header with an optional field: read as it comes, then its payload, however the bits are cut"""
    h = _config.load_obj("header", "t", {"base": "header", "type": "link", "fields": [
        {"name": "ext", "bits": 1}, {"name": "len", "bits": 7}, {"name": "more", "bits": 8, "if": "ext == 1"}],
        "payload": "8*len", "crc": "crc8_smbus"})
    frames = [(0, b"abc"), (1, b"hello world"), (0, b"z")]
    tx = []
    for ext, p in frames:
        h.control(frame=True, ext=ext, len=len(p), more=0x5A)
        tx.append(np.asarray((rp.Data(p, flush=True) * h).data).astype(bool))
    bits = np.concatenate(tx)
    assert len(tx[1]) == 8 + 8 + 8 + 8*11 and len(tx[0]) == 8 + 8 + 8*3 # the optional byte only where ext is set
    rx = _config.load_obj("header", "t2", {"base": "header", "type": "link", "fields": [
        {"name": "ext", "bits": 1}, {"name": "len", "bits": 7}, {"name": "more", "bits": 8, "if": "ext == 1"}],
        "payload": "8*len", "crc": "crc8_smbus"})
    got, ctx, at = [], {}, 0
    # One frame per call on receive here (a profile's delimiter or a framed length starts each one): feed each frame cut up
    for f in tx:
        rx.control(frame=True)
        cuts = np.sort(rng.choice(np.arange(1, len(f)), 3, replace=False))
        for part in np.split(f, cuts):
            o = rp.Data(part, ctx=ctx) / rx
            ctx = o.ctx
            got.append(np.asarray(o.data).astype(bool))
    assert np.packbits(np.concatenate(got).astype(np.uint8)).tobytes() == b"".join(p for _, p in frames)

def test_ax25_frame_message_fields():
    fr = rp.link.load("frame/ax25")
    def addr(call, ssid, last):
        return bytes(c << 1 for c in call.ljust(6).encode()) + bytes([0x60 | ssid << 1 | last])
    pkt = addr("APRS", 0, 0) + addr("N0CALL", 7, 0) + addr("WIDE1", 1, 1) + b"\x03\xf0" + b"!4903.50N/07201.75W-"
    bits = np.asarray((rp.Data(pkt) * fr).data)
    from rpps.base.control import emit # noqa: F401 (messages go out through emit; read them from a profile runner)
    p = rp.profile.load("ax25/afsk1200")
    assert any(s.spec.get("name") == "ax25" for s in p.steps)
    v = fr.payload_schema.parse(np.unpackbits(np.frombuffer(pkt, np.uint8)).astype(bool))[0]
    assert [(a["call"], a["ssid"]) for a in v["addresses"]] == [("APRS", 0), ("N0CALL", 7), ("WIDE1", 1)]
    assert v["control"] == 3 and v["pid"] == 0xF0 and v["pid.name"] == "no layer 3"
    assert bytes.fromhex((rp.Data(bits, flush=True) / fr).as_bytes().hex) == pkt

def test_ble_adv_payload_fields():
    h = rp.link.load("header/ble_adv")
    payload = bytes.fromhex("010203040506" "020106" "0409414243") # AdvA, Flags 0x06, Complete Local Name "ABC"
    v = h.payload_values(np.unpackbits(np.frombuffer(payload, np.uint8)).astype(bool), {"type": 2})
    assert v["adva"] == bytes.fromhex("060504030201")
    assert v["ad"] == [{"len": 2, "ad_type": 1, "ad_type.name": "Flags", "data": b"\x06"},
                       {"len": 4, "ad_type": 9, "ad_type.name": "Name", "data": b"ABC"}]
    assert "ad" not in h.payload_values(np.unpackbits(np.frombuffer(payload[:6], np.uint8)).astype(bool), {"type": 3})

def test_ts_adaptation_field_from_the_profile():
    spec = next(s for s in rp.profile.load("dvbs/qpsk").obj["steps"] if "message" in s)["message"]
    s = Schema(spec["fields"])
    pcr = 0x123456789 & ((1 << 33) - 1)
    hdr = bits_of((0x47, 8), (0, 1), (1, 1), (0, 1), (0x100, 13), (0, 2), (3, 2), (5, 4))
    ad = bits_of((7, 8), (0x10, 8), (pcr, 33), (0x3F, 6), (17, 9))
    v = s.parse(np.concatenate((hdr, ad, np.zeros(8*(188 - 12), bool))))[0]
    assert v["pid"] == 0x100 and v["pusi"] == 1 and v["afc"] == 3 and v["cc"] == 5
    assert v["adaptation"] == {"length": 7, "flags": 0x10, "pcr_base": pcr, "pcr_reserved": 0x3F, "pcr_ext": 17}

def test_sized_group_align_tx_counts_enum_names_and_bit_values():
    s = Schema([{"name": "n", "bits": 8, "tx": "len(items) - 1"},
                {"name": "items", "repeat": "n + 1", "fields": [{"name": "v", "bits": 4}]},
                {"name": "len", "bits": 8, "tx": "len(fmt)"},
                {"name": "fmt", "bytes": "len", "fields": [
                    {"name": "m", "bits": 3, "enum": {"5": "five"}}, {"name": "pad", "align": 8},
                    {"name": "wl", "bits": 4, "tx": "len(w)"}, {"name": "w", "bits": "wl"}]},
                {"name": "end", "bits": 8}])
    bits = s.build({"items": [{"v": 1}, {"v": 2}], "fmt": {"m.name": "five", "w": [1, 0, 1]}, "end": 0xAB})
    # n=1, items 1 and 2, len 2, fmt: m=5 then 5 stuffing bits, wl=3, w=101, padded to its 2 bytes, then end
    assert np.array_equal(bits, bits_of((1, 8), (1, 4), (2, 4), (2, 8), (0b10100000, 8), (0b00111010, 8), (0xAB, 8)))
    v, used = s.parse(bits)
    assert used == len(bits) and v["end"] == 0xAB and v["items"] == [{"v": 1}, {"v": 2}]
    assert v["fmt"] == {"m": 5, "m.name": "five", "wl": 3, "w": 0b101}
    assert np.array_equal(s.build(v), bits) # (counts given now: the same)
    with pytest.raises(Short):
        s.parse(bits[:30]) # (the sized group's bytes not all there)
    with pytest.raises(ConfigError, match="no value named"):
        s.build({"fmt": {"m.name": "six"}})
    with pytest.raises(ConfigError, match="align"):
        Schema([{"name": "pad", "align": 8, "bits": 3}])

# ---- link: a frame check sequence after the payload, and streaming HDLC

def test_header_fcs_round_trip_and_rejects_a_damaged_frame(rng):
    msg = rng.integers(0, 2, 8*20).astype(bool)
    tx = rp.Data(msg) * rp.link.load("header/ble_adv")
    assert len(tx) == 16 + 160 + 24
    out = as_data_bits(np.asarray(tx.data)) / rp.link.load("header/ble_adv")
    assert np.array_equal(hard(out), msg) and out.stream.est["ble_adv"]["last"]["length"] == 20
    bad = np.asarray(tx.data).copy()
    bad[16 + 50] ^= True
    out = as_data_bits(bad) / rp.link.load("header/ble_adv")
    assert len(out) == 0 and out.stream.est["ble_adv"]["fcs_failed"] == 1 and out.stream.est["ble_adv"]["frames"] == 0

def test_mode_s_header_matches_a_real_frame():
    """A real DF17 extended squitter (ICAO address 4840D6): the header reads its DF, checks its parity and gives the 83 bits between"""
    frame = bytes.fromhex("8D4840D6202CC371C32CE0576098")
    bits = np.unpackbits(np.frombuffer(frame, np.uint8)).astype(bool)
    out = as_data_bits(bits) / rp.link.load("header/mode_s")
    assert out.stream.est["mode_s"]["last"] == {"df": 17} and len(out) == 83
    assert np.array_equal(hard(out), bits[5:88])

# ---- BCD, MJD dates, BCD times, DVB text, lists filling bytes, shared groups (EN 300 468)

def test_bcd_mjd_and_time():
    s = Schema([{"name": "f", "type": "bcd", "bits": 32, "scale": 10000}, {"name": "d", "type": "mjd", "bits": 16},
                {"name": "t", "type": "time", "bits": 24}, {"name": "o", "type": "time", "bits": 16}])
    # EN 300 468 annex C: 93/10/13 12:45:00 is MJD 0xC079, BCD 0x124500; 11.778 GHz is 0x01177800 in 10 kHz digits
    b = bits_of((0x01177800, 32), (0xC079, 16), (0x124500, 24), (0x0130, 16))
    v, used = s.parse(b)
    assert v == {"f": 11778000000, "d": "1993-10-13", "t": "12:45:00", "o": "01:30"} and used == 88
    assert np.array_equal(s.build(v), b)
    assert s.fixed_bits == 88

def test_dvb_text_tables():
    from rpps.link.fields import dvb_text, dvb_bytes
    assert dvb_text(b"News") == "News"
    assert dvb_text(b"\x15Caf\xc3\xa9") == "Café"
    assert dvb_text(b"\x05\xc4\xe9") == "Äé" # 0x05: ISO 8859-9
    assert dvb_text(b"\x10\x00\x02\xa1") == "Ą" # 0x10 0x00 n: ISO 8859-n
    assert dvb_text(b"\x11\x04\x14") == "Д" # UCS-2
    assert dvb_text(b"A\x86B\x87\x8aC") == "AB\nC" # emphasis codes go, 0x8A breaks the line
    assert dvb_bytes("News") == b"News" and dvb_bytes("Café") == b"\x15Caf\xc3\xa9"
    s = Schema([{"name": "n", "bits": 8, "tx": "len(name)"}, {"name": "name", "type": "text", "encoding": "dvb", "bytes": "n"}])
    b = s.build({"name": "Café"})
    assert s.parse(b)[0] == {"n": 6, "name": "Café"}

def test_lists_filling_bytes_groups_and_the_rest():
    groups = {"d": [{"name": "tag", "bits": 8}, {"name": "length", "bits": 8, "tx": "len(body)"},
                    {"name": "body", "bytes": "length", "fields": [{"name": "data", "type": "bytes", "bytes": "*"}]}]}
    s = Schema([{"name": "n", "bits": 8, "tx": "len(ds)"}, {"name": "ds", "bytes": "n", "repeat": "*", "group": "d"},
                {"name": "after", "bits": 8}], groups=groups)
    v = {"ds": [{"tag": 1, "body": {"data": b"ab"}}, {"tag": 2, "body": {"data": b""}}], "after": 9}
    b = s.build(v)
    assert np.packbits(b).tobytes() == bytes([6, 1, 2]) + b"ab" + bytes([2, 0, 9])
    got, used = s.parse(b)
    assert got["ds"] == [{"tag": 1, "length": 2, "body": {"data": b"ab"}}, {"tag": 2, "length": 0, "body": {"data": b""}}]
    assert got["after"] == 9 and used == len(b)
    with pytest.raises(ConfigError):
        Schema([{"name": "x", "group": "g"}], groups={"g": [{"name": "y", "group": "g"}]})
    with pytest.raises(ConfigError):
        Schema([{"name": "x", "group": "nope"}])
    with pytest.raises(ConfigError):
        Schema([{"name": "x", "bytes": 2, "repeat": 3, "fields": [{"name": "y", "bits": 8}]}])
