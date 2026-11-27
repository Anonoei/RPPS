"""Blind CRC search on frames (rpps.analysis.framing.crc)"""
import numpy as np
import pytest

import rpps as rp
from rpps.analysis.framing import crc as C

def crc_of(name):
    return rp.link.load(f"crc/{name}").encoder

def frames(rng, code, lengths, n, head=None, head_vary=0, trail=0, trail_vary=False):
    """n frames of code over random messages (lengths in turn); a header (constant, its first head_vary bits random
    per frame) before, `trail` bits after (constant ones, or random)"""
    out = []
    for i in range(n):
        h = np.array(head if head is not None else [], dtype=bool)
        h[:head_vary] = rng.integers(0, 2, head_vary)
        t = rng.integers(0, 2, trail).astype(bool) if trail_vary else np.ones(trail, bool)
        out.append(np.concatenate((h, code.encode(rng.integers(0, 2, lengths[i % len(lengths)]).astype(bool)), t)))
    return out

@pytest.mark.parametrize("name, lengths", [("crc8_smbus", [64]), ("crc12_p25", [72]), ("crc16_ccitt_false", [96]),
                                           ("crc24_ble", [120]), ("crc32_mpeg2", [400]), ("crc32_ieee", [200, 256, 304])])
def test_shipped_crcs_named(name, lengths):
    rng = np.random.default_rng(1)
    F = frames(rng, crc_of(name), lengths, 14)
    f = C.search(F)
    assert f is not None and name in f.names and f.width == crc_of(name).width
    assert f.start == 0 and f.tail == 0 and f.hits == len(F)
    assert f.ambiguous == (len(lengths) == 1)
    assert all(f.check(x) for x in F)

def test_init_and_xorout_from_two_lengths():
    """A CRC no definition has: init and xorout come out exactly from two message lengths"""
    rng = np.random.default_rng(2)
    log = rp.link.load("crc/crc16_arc").log
    code = rp.link.crc(log, width=16, poly=0x002D, init=0x1D0F, xorout=0x5A5A) # x^16 + x^5 + x^3 + x^2 + 1, primitive
    F = frames(rng, code, [80, 112], 16)
    f = C.search(F)
    assert f is not None and not f.names and not f.ambiguous
    assert (f.width, f.poly, f.init, f.xorout) == (16, 0x002D, 0x1D0F, 0x5A5A)

def test_generator_with_x_plus_1_leaves_init_open():
    """x + 1 divides both P and x^n1 + x^n2: one init bit stays open, and every frame still passes"""
    rng = np.random.default_rng(2)
    code = rp.link.crc(rp.link.load("crc/crc16_arc").log, width=16, poly=0x8005, init=0x1D0F, xorout=0x5A5A)
    F = frames(rng, code, [80, 112], 16)
    f = C.search(F)
    assert f is not None and (f.width, f.poly) == (16, 0x8005) and f.ambiguous and f.hits == len(F)
    assert all(f.check(x) for x in F)

def test_guess_saves_and_loads(tmp_path):
    rng = np.random.default_rng(3)
    code = rp.link.crc(rp.link.load("crc/crc16_arc").log, width=10, poly=0x233, init=0x3FF, xorout=0)
    F = frames(rng, code, [70, 90], 12)
    f = C.search(F)
    assert f is not None and (f.width, f.poly) == (10, 0x233)
    f.guess.save("found", path=tmp_path)
    loaded = rp.link.load("crc/guess/found", path=tmp_path).encoder
    assert all(loaded.syndrome(x) == 0 for x in F)

def test_header_fill_and_trailer():
    """Coverage after a constant header; the field before constant fill, or before bits that vary"""
    rng = np.random.default_rng(4)
    code = crc_of("crc16_xmodem")
    f = C.search(frames(rng, code, [96], 12, head=[1, 0]*8, trail=6))
    assert f is not None and "crc16_xmodem" in f.names and (f.start, f.tail) == (16, 6)
    f = C.search(frames(rng, code, [200], 12, trail=20, trail_vary=True))
    assert f is not None and "crc16_xmodem" in f.names and f.tail == 20

def test_uncovered_header_that_varies():
    """An address / counter before the covered bits: the differences are combined to cancel it"""
    rng = np.random.default_rng(5)
    F = frames(rng, crc_of("crc16_ccitt_false"), [200], 80, head=np.zeros(40), head_vary=40)
    f = C.search(F)
    assert f is not None and "crc16_ccitt_false" in f.names and f.start == 40 and f.hits == 80

def test_some_frames_corrupted():
    rng = np.random.default_rng(6)
    F = frames(rng, crc_of("crc16_ccitt_false"), [200], 20)
    for x in F[:3]:
        x[rng.integers(len(x))] ^= True
    f = C.search(F)
    assert f is not None and "crc16_ccitt_false" in f.names and f.hits == 17

def test_bytes_in():
    rng = np.random.default_rng(7)
    F = [np.packbits(x) for x in frames(rng, crc_of("crc32_mpeg2"), [96], 10)]
    f = C.search(F)
    assert f is not None and "crc32_mpeg2" in f.names

def test_no_crc_in_random_frames():
    """Measured: 0 of 100 sets of 30 random 160-bit frames give a CRC; 40 sets here"""
    hits = sum(C.search([r.integers(0, 2, 160).astype(bool) for _ in range(30)]) is not None
               for r in (np.random.default_rng(100 + s) for s in range(40)))
    assert hits == 0
