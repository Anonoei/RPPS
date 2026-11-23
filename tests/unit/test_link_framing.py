"""Checksums (rpps.link.checksum) and byte-stuffed framing (rpps.link.bytestuff: SLIP, KISS, asynchronous PPP)"""
import numpy as np
import pytest

import rpps as rp
from rpps import _config
from rpps.errors import ConfigError

def checksum(**o):
    return _config.load_obj("crc", "t", {"base": "crc", "type": "checksum", **o}).encoder

def test_published_check_values():
    assert checksum(algorithm="fletcher", width=16).compute(b"abcde") == 0xC8F0            # Wikipedia, Fletcher's checksum
    assert checksum(algorithm="fletcher", width=16).compute(b"abcdef") == 0x2057
    assert checksum(algorithm="fletcher", width=32, endian="little").compute(b"abcde") == 0xF04FC729
    assert checksum(algorithm="adler").compute(b"Wikipedia") == 0x11E60398                 # zlib's adler32
    assert checksum(algorithm="internet").compute(bytes.fromhex("0001f203f4f5f6f7")) == 0x220D # RFC 1071 3
    assert checksum(algorithm="sum", negate=True).compute(bytes.fromhex("0300300002337a")) == 0x1E # an Intel HEX record
    assert checksum(algorithm="xor").compute(b"GPGLL,4916.45,N,12311.12,W,225444,A") == 0x31   # an NMEA sentence

def test_checksum_has_the_crc_interface(rng):
    for c in (checksum(algorithm="internet"), checksum(algorithm="fletcher", width=16, refin=True, order="little"),
              checksum(algorithm="sum", width=16, word=8)):
        m = bytes(rng.integers(0, 256, 21).astype(np.uint8))
        bits = np.unpackbits(np.frombuffer(m, np.uint8), bitorder="little" if c.refin else "big").astype(bool)
        assert c.compute_bits(bits) == c.compute(m)
        cw = c.encode(bits)
        assert c.syndrome(cw) == 0 and c.decode(cw, status=True)[1][0] == 0
        bad = cw.copy()
        bad[3] ^= True
        assert c.decode(bad, status=True)[1][0] == -1

def test_checksum_definition_errors():
    with pytest.raises(ValueError, match="check value"):
        checksum(algorithm="adler", check="0x1")
    with pytest.raises(ConfigError, match="algorithm"):
        checksum(algorithm="md5")
    with pytest.raises(ConfigError, match="16 or 32"):
        checksum(algorithm="fletcher", width=8)

def test_a_checksum_as_a_composite_part(tmp_path):
    import json
    (tmp_path/"crc").mkdir()
    (tmp_path/"crc"/"t_xor8.json").write_text(json.dumps({"base": "crc", "type": "checksum", "comment": "t",
                                                          "algorithm": "xor"}))
    rp.add_config_path(tmp_path)
    try:
        c = _config.load_obj("crc", "t", {"base": "crc", "type": "composite",
                                          "parts": [{"crc": "crc8_smbus"}, {"crc": "t_xor8", "input": ["nibble_swap"]}]}).encoder
        m = b"\x12\x34\x56"
        assert c.compute(m) == rp.link.load("crc/crc8_smbus").encoder.compute(m) << 8 | (0x21 ^ 0x43 ^ 0x65)
    finally:
        _config._EXTRA.remove(tmp_path)

def test_slip_kiss_and_ppp_escape_as_their_rfcs_say():
    data = bytes([0x00, 0xC0, 0xDB, 0x7E, 0x7D, 0x11, 0x41])
    slip = rp.link.load("frame/slip").encoder
    assert slip._bytes(slip.encode([data])).hex() == "c000dbdcdbdd7e7d1141c0" # RFC 1055: END ESC_END, ESC ESC_ESC
    ppp = rp.link.load("frame/ppp_async").encoder
    sent = ppp._bytes(ppp.encode([data]))
    assert sent[:12].hex() == "7e7d20c0db7d5e7d5d7d3141" # RFC 1662: 7D then the byte XOR 0x20 for 00, 7E, 7D, 11
    fcs = rp.link.load("crc/crc16_x25").encoder
    assert fcs.compute(data + fcs.to_bytes(fcs.compute(data))) == 0x0F47 # the good-FCS residue (0xF0B8, reflected)
    assert rp.link.load("frame/kiss").encoder.frames_at(slip.encode([b"\x00ax25"]))[0][0][1] == b"\x00ax25"

@pytest.mark.parametrize("name", ["slip", "kiss", "ppp_async"])
def test_frames_any_block_cut(rng, name):
    fr = rp.link.load(f"frame/{name}")
    frames = [bytes(rng.integers(0, 256, n).astype(np.uint8)) for n in (3, 40, 1, 200, 17)]
    frames = [f for f in frames if len(f) >= fr.encoder.min_bytes]
    bits = np.concatenate([np.asarray((rp.Data(f) * fr).data) for f in frames]).astype(bool)
    want = b"".join(frames)
    assert bytes.fromhex((rp.Data(bits, flush=True) / fr).as_bytes().hex) == want
    cuts = np.sort(rng.choice(np.arange(1, len(bits)), 9, replace=False))
    ctx, out = {}, b""
    parts = np.split(bits, cuts)
    for k, part in enumerate(parts): # (a Pipe keeps what it holds in ctx)
        o = rp.Data(part, ctx=ctx, flush=k == len(parts) - 1) / fr
        ctx = o.ctx
        out += bytes.fromhex(o.as_bytes().hex) if len(o) else b""
    assert out == want

def test_a_bad_fcs_and_a_bad_escape():
    ppp = rp.link.load("frame/ppp_async").encoder
    good, bad = b"\xff\x03\x00\x21hello", b"\xff\x03\x00\x21world"
    raw = bytearray(ppp._bytes(ppp.encode([good, bad])))
    raw[-3] ^= 0x01 # in the second frame's FCS (no byte there needs escaping)
    frames, aborts = ppp.frames_at(ppp._bits(bytes(raw)))
    assert [(p, ok) for _, p, ok in frames] == [(good, True), (bad, False)] and aborts == 0
    slip = rp.link.load("frame/slip").encoder
    frames, aborts = slip.frames_at(slip._bits(bytes.fromhex("c00102dbffc0c00304c0")))
    assert [p for _, p, _ in frames] == [b"\x03\x04"] and aborts == 1 # ESC before a byte it doesn't escape

def test_uart_bit_order():
    c = _config.load_obj("frame", "t", {"base": "frame", "type": "bytes", "flag": "0xC0", "escape": "0xDB",
                                        "escapes": {"0xC0": "0xDC", "0xDB": "0xDD"}, "refin": True}).encoder
    bits = c.encode([b"\x01"])
    assert np.array_equal(bits[8:16].astype(int), [1, 0, 0, 0, 0, 0, 0, 0]) # the byte least significant bit first
    assert c.frames_at(bits)[0][0][1] == b"\x01"
