"""Composite checks (rpps.link.check): CRC parts over transformed views of the message, wherever a CRC goes"""
import numpy as np
import pytest

import rpps as rp
from rpps import _config
from rpps.errors import ConfigError
from rpps.link.hdlc import hdlc
from rpps.logger import Logger

def load(parts, **kw):
    return _config.load_obj("crc", "t", {"base": "crc", "type": "composite", "parts": parts, **kw}).encoder

def swap(b):
    return bytes(((x << 4) | (x >> 4)) & 0xFF for x in b)

CRC8 = "crc8_smbus"
SPLIT = [{"crc": CRC8}, {"crc": CRC8, "input": ["nibble_swap"]}]

def test_concat_is_the_parts_in_order(rng):
    c, c8 = load(SPLIT), rp.link.load(f"crc/{CRC8}").encoder
    for n in (0, 1, 7, 40):
        m = bytes(rng.integers(0, 256, n).astype(np.uint8))
        assert c.width == 16
        assert c.compute(m) == (c8.compute(m) << 8) | c8.compute(swap(m))
        assert c.to_bytes(c.compute(m)) == bytes([c8.compute(m), c8.compute(swap(m))])
        bits = np.unpackbits(np.frombuffer(m, np.uint8)).astype(bool)
        assert c.compute_bits(bits) == c.compute(m)
        assert c.from_bits(c.field(bits)) == c.compute(m)

def test_encode_decode_and_a_flipped_bit(rng):
    c = load(SPLIT)
    bits = rng.integers(0, 2, 80).astype(bool)
    cw = c.encode(bits)
    msg, st = c.decode(cw, status=True)
    assert np.array_equal(msg, bits) and st[0] == 0 and c.syndrome(cw) == 0
    for i in range(len(cw)):
        bad = cw.copy()
        bad[i] ^= True
        assert c.decode(bad, status=True)[1][0] == -1

def test_reflected_parts_read_bits_the_way_they_are_sent(rng):
    c, x25 = load([{"crc": "crc16_x25", "input": ["reverse_bytes"]}]), rp.link.load("crc/crc16_x25").encoder
    m = bytes(rng.integers(0, 256, 12).astype(np.uint8))
    assert c.compute(m) == x25.compute(m[::-1])
    bits = np.unpackbits(np.frombuffer(m, np.uint8), bitorder="little") # refin: each byte least significant bit first
    assert c.compute_bits(bits) == c.compute(m)

def test_xor_transforms_and_outputs():
    m = b"123456789"
    c8 = rp.link.load(f"crc/{CRC8}").encoder
    c = load([{"crc": CRC8}, {"crc": CRC8, "input": [{"op": "xor", "value": ["0x5a", 1]}, {"op": "skip", "n": 1}],
                              "output": ["invert"]}], combine="xor")
    pat = np.resize(np.array([0x5A, 1], np.uint8), len(m))
    alt = (np.frombuffer(m, np.uint8) ^ pat)[1:].tobytes()
    assert c.width == 8 and c.compute(m) == c8.compute(m) ^ c8.compute(alt) ^ 0xFF
    with pytest.raises(ConfigError, match="one width"):
        load([{"crc": CRC8}, {"crc": "crc16_x25"}], combine="xor")

def test_check_value_and_unknown_transform():
    good = load(SPLIT).compute(b"123456789")
    load(SPLIT, check=hex(good))
    with pytest.raises(ValueError, match="check value"):
        load(SPLIT, check=hex(good ^ 1))
    with pytest.raises(ConfigError, match="no transform"):
        load([{"crc": CRC8, "input": ["nope"]}])

def test_registered_transform():
    from rpps.link import check
    check.register("_test_add1")(lambda d: (d + 1).astype(np.uint8))
    try:
        c8 = rp.link.load(f"crc/{CRC8}").encoder
        assert load([{"crc": CRC8, "input": ["_test_add1"]}]).compute(b"\x00\x10") == c8.compute(b"\x01\x11")
    finally:
        del check.TRANSFORMS["_test_add1"]

def test_byte_transforms_need_whole_bytes():
    with pytest.raises(ValueError, match="whole bytes"):
        load(SPLIT).compute_bits(np.ones(13, bool))

def test_hdlc_with_a_composite_fcs_any_cut(rng):
    code = hdlc(Logger().Child("t", 0), load(SPLIT))
    frames = [bytes(rng.integers(0, 256, n).astype(np.uint8)) for n in (5, 30, 200, 9)]
    bits = code.encode(frames)
    assert [p for p, ok in code.deframe(bits)[0] if ok] == frames
    out, held = [], np.zeros(0, np.uint8)
    cuts = np.sort(rng.choice(len(bits), 12, replace=False))
    for part in np.split(bits, cuts):
        o, held = code.decode_stream(part, held)
        out.append(o)
    out.append(code.decode(held))
    got = np.packbits(np.concatenate(out).astype(np.uint8)).tobytes()
    assert got == b"".join(frames)
    bad = bits.copy()
    bad[40] ^= True # inside the first frame: its FCS fails (or the flip breaks the stuffing and it aborts)
    assert [p for p, ok in code.deframe(bad)[0] if ok] == frames[1:]
