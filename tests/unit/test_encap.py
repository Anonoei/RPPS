import numpy as np
import pytest

import rpps as rp
from conftest import bits

# Catalogue check values for b"123456789"
CHECK = {
    "crc32_ieee": 0xCBF43926, "crc32_mpeg2": 0x0376E6E7,
    "crc32c": 0xE3069283, "crc32_bzip2": 0xFC891918, "crc16_kermit": 0x2189, "crc16_modbus": 0x4B37, "crc8_smbus": 0xF4,
    "crc16_arc": 0xBB3D, "crc16_xmodem": 0x31C3, "crc16_ccitt_false": 0x29B1, "crc16_x25": 0x906E,
    "crc24_ble": 0xC25A56, "crc24_lte_a": 0xCDE703, "crc24_lte_b": 0x23EF52, "crc16_gsm": 0xCE3C, "crc16_genibus": 0xD64E,
}

@pytest.mark.parametrize("name", rp.encap.ls()["crc"])
def test_crc_loads_and_self_checks(name):
    crc = rp.encap.load("crc", name)._enc # A wrong definition raises on load (its own check value)
    assert crc.compute(b"123456789") == crc.check

@pytest.mark.parametrize("name,value", CHECK.items())
def test_crc_catalogue_values(name, value):
    assert rp.encap.load("crc", name)._enc.compute(b"123456789") == value

@pytest.mark.parametrize("name", rp.encap.ls()["crc"])
@pytest.mark.parametrize("n", [8*40, 37, 1])
def test_crc_append_and_detect(name, n, rng):
    crc = rp.encap.load("crc", name)._enc
    msg = bits(rng, n)
    coded = crc.encode(msg)
    assert len(coded) == len(msg) + crc.width
    assert np.array_equal(crc.decode(coded), msg) and crc.ok
    bad = coded.copy()
    bad[min(17, len(bad) - 1)] ^= True
    crc.decode(bad)
    assert not crc.ok

def _serial(bits, width, poly, init=0, xorout=0):
    """Bit-at-a-time CRC as the standards draw it (no table, no reflection)"""
    reg, top = init, 1 << (width - 1)
    for b in bits:
        fb = bool(reg & top) ^ bool(b)
        reg = (reg << 1) & (2*top - 1)
        if fb:
            reg ^= poly
    return reg ^ xorout

@pytest.mark.parametrize("name", rp.encap.ls()["crc"])
def test_crc_bytes_and_bits_agree(name, rng):
    crc = rp.encap.load("crc", name)._enc
    data = rng.integers(0, 256, 23).astype(np.uint8)
    assert crc.compute(data) == crc.compute_bits(np.unpackbits(data, bitorder="little" if crc.refin else "big"))

@pytest.mark.parametrize("width,poly,init,xorout,ref", [
    (16, 0x1021, 0, 0, "crc16_xmodem"), (24, 0x864CFB, 0, 0, "crc24_lte_a"), (16, 0x1021, 0xFFFF, 0xFFFF, "crc16_genibus"),
    (12, 0x897, 0, 0xFFF, "crc12_p25"), (24, 0xFFF409, 0, 0, "crc24_mode_s")])
def test_crc_any_length_matches_serial(width, poly, init, xorout, ref, rng):
    crc = rp.encap.load("crc", ref)._enc
    for n in (1, 7, 13, 21, 24, 37, 100):
        msg = bits(rng, n)
        assert crc.compute_bits(msg) == _serial(msg, width, poly, init, xorout)

@pytest.mark.parametrize("width,poly,init,refin,refout,xorout,check", [
    (3, 0x3, 0, False, False, 0x7, 0x4),          # CRC-3/GSM
    (5, 0x05, 0x1F, True, True, 0x1F, 0x19),      # CRC-5/USB
    (6, 0x27, 0x3F, False, False, 0, 0x0D),       # CRC-6/CDMA2000-A
    (12, 0x80F, 0, False, False, 0, 0xF5B),       # CRC-12/DECT
    (40, 0x0004820009, 0, False, False, 0xFFFFFFFFFF, 0xD4164FC646), # CRC-40/GSM
])
def test_crc_catalogue_odd_widths(width, poly, init, refin, refout, xorout, check, rng):
    crc = rp.encap.crc(rp.logger.Logger(), width, poly, init, refin, refout, xorout, check)
    msg = bits(rng, 29)
    assert np.array_equal(crc.decode(crc.encode(msg)), msg) and crc.ok

def test_crc_ble_against_spec_lfsr(rng):
    """Core spec Vol 6 Part B 3.1.1: a 24-stage LFSR, position 0 nearest the input, CRC sent from position 23"""
    def spec(bits, init=0x555555):
        p = [(init >> i) & 1 for i in range(24)]
        for d in bits:
            fb = int(d) ^ p[23]
            p = [fb] + p[:23]
            for t in (1, 3, 4, 6, 9, 10):
                p[t] ^= fb
        return [p[i] for i in range(23, -1, -1)]
    crc = rp.encap.load("crc", "crc24_ble")._enc
    for n in (16, 48, 8*37):
        msg = bits(rng, n)
        assert list(crc.encode(msg)[n:].astype(int)) == spec(msg)

def test_crc_mode_s_frame():
    crc = rp.encap.load("crc", "adsb")._enc
    frame = np.unpackbits(np.frombuffer(bytes.fromhex("8D4840D6202CC371C32CE0576098"), np.uint8)) # DF17, KLM1023
    assert crc.syndrome(frame) == 0
    crc.decode(frame)
    assert crc.ok
    frame[40] ^= 1
    assert crc.syndrome(frame) != 0

def test_crc_mask_overlay(rng):
    """LTE BCH: the CRC is XORed with an antenna-port mask; the syndrome of an unmasked check recovers it"""
    crc = rp.encap.load("crc", "crc16_xmodem")._enc
    msg = bits(rng, 24)
    sent = crc.encode(msg, mask=0xFFFF)
    assert crc.syndrome(sent) == 0xFFFF
    crc.decode(sent, mask=0xFFFF)
    assert crc.ok
    crc.decode(sent)
    assert not crc.ok

def test_crc_width_out_of_range():
    with pytest.raises(ValueError):
        rp.encap.crc(rp.logger.Logger(), 65, 1)

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
