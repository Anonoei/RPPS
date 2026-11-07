import numpy as np
import pytest

import rpps as rp
from conftest import bits

# Catalogue check values for b"123456789"
CHECK = {
    "crc32_ieee": 0xCBF43926, "crc32_mpeg2": 0x0376E6E7,
    "crc32c": 0xE3069283, "crc32_bzip2": 0xFC891918, "crc16_kermit": 0x2189, "crc16_modbus": 0x4B37, "crc8_smbus": 0xF4,
    "crc16_arc": 0xBB3D, "crc16_xmodem": 0x31C3, "crc16_ccitt_false": 0x29B1, "crc16_x25": 0x906E,
}

@pytest.mark.parametrize("name", rp.encap.ls()["crc"])
def test_crc_loads_and_self_checks(name):
    crc = rp.encap.load("crc", name)._enc # A wrong definition raises on load (its own check value)
    assert crc.compute(b"123456789") == crc.check

@pytest.mark.parametrize("name,value", CHECK.items())
def test_crc_catalogue_values(name, value):
    assert rp.encap.load("crc", name)._enc.compute(b"123456789") == value

@pytest.mark.parametrize("name", rp.encap.ls()["crc"])
def test_crc_append_and_detect(name, rng):
    crc = rp.encap.load("crc", name)._enc
    msg = bits(rng, 8*40)
    coded = crc.encode(msg)
    assert len(coded) == len(msg) + crc.width
    assert np.array_equal(crc.decode(coded), msg) and crc.ok
    bad = coded.copy()
    bad[17] ^= True
    crc.decode(bad)
    assert not crc.ok

@pytest.mark.parametrize("name", rp.encap.ls()["frame"])
def test_hdlc_round_trip(name, rng):
    h = rp.encap.load("frame", name)._enc
    frames = [bytes(rng.integers(0, 256, n).astype(np.uint8)) for n in (10, 33, 100)]
    frames.append(b"\xff"*12) # Runs of ones need bit stuffing
    got, aborts = h.deframe(h.encode(frames))
    assert aborts == 0
    assert [p for p, ok in got] == frames
    assert all(ok for p, ok in got)

@pytest.mark.parametrize("name", rp.encap.ls()["frame"])
def test_hdlc_bad_fcs_flagged(name, rng):
    h = rp.encap.load("frame", name)._enc
    frames = [bytes(rng.integers(0, 256, 40).astype(np.uint8)) for _ in range(3)]
    bits = h.encode(frames).copy()
    mid = len(bits)//2
    bits[mid] ^= True
    h.decode(bits)
    good, bad, aborts = h.last
    assert len(good) < len(frames) and (bad + aborts) >= 1
