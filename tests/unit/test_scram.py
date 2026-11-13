import numpy as np
import pytest

import rpps as rp
from conftest import bits

ALL = [(k, n) for k, names in rp.code.scram.ls().items() for n in names]

def _load(folder, name):
    """With example values for the parameters a definition needs (a cell id, a WACN ...)"""
    return rp._config.load("scramble", name, folder, example=True)

@pytest.mark.parametrize("folder,name", ALL)
def test_scram_round_trip(folder, name, rng):
    s = _load(folder, name)
    msg = bits(rng, 8*204*8) # Whole frames for the frame-based scramblers
    scr = s.encode(msg)
    assert not np.array_equal(scr, msg)
    assert np.array_equal(np.asarray(s.decode(scr)).astype(bool), msg)

@pytest.mark.parametrize("folder,name", ALL)
def test_scram_whitens_constant_input(folder, name):
    s = _load(folder, name)
    scr = np.asarray(s.encode(np.zeros(8*204*8, dtype=bool)))
    assert 0.4 < scr.mean() < 0.6

@pytest.mark.parametrize("name", ["v35", "galois", "fib", "g3ruh"])
def test_fdt_keystream_follows_polynomial(name):
    """The keystream is an LFSR sequence: s[n-N] = s[n] ^ (xor of s[n-t] for the other taps t)"""
    s = rp.code.scram.load(name)
    taps = [int(t) for t in s.s_lfsr._poly] if hasattr(s.s_lfsr, "_poly") else [int(t) for t in s.s_lfsr.taps]
    N = max(taps)
    y = np.asarray(s.encode(np.zeros(2000, dtype=bool))).astype(int)
    assert 0.45 < y.mean() < 0.55
    for i in range(N, len(y)):
        assert y[i-N] == y[i] ^ np.bitwise_xor.reduce([y[i-t] for t in taps if t < N])

FDT_DELAYS = {"g3ruh": (12, 17), "v35": (3, 20), "galois": (11, 13, 14, 16), "fib": (11, 13, 14, 16)}

@pytest.mark.parametrize("name", FDT_DELAYS)
def test_fdt_is_self_synchronizing(name, rng):
    """The scrambled bits feed the register: y[n] = d[n] ^ y[n-t], descrambled d[n] = y[n] ^ y[n-t]; a descrambler that
    joins mid-stream (wrong register) is right after one register length; one channel error is one error per tap plus itself"""
    delays, N = FDT_DELAYS[name], max(FDT_DELAYS[name])
    d = bits(rng, 600)
    y = np.asarray(rp.code.scram.load(name).encode(d)).astype(int)
    for n in range(N, len(y)):
        assert y[n] == int(d[n]) ^ np.bitwise_xor.reduce([y[n - t] for t in delays])
    assert np.array_equal(np.asarray(rp.code.scram.load(name).decode(y.astype(bool))).astype(bool), d)
    mid = np.asarray(rp.code.scram.load(name).decode(y[137:].astype(bool))).astype(bool)
    assert np.array_equal(mid[N:], d[137 + N:])
    bad = y.copy()
    bad[300] ^= 1
    ref = np.asarray(rp.code.scram.load(name).decode(y.astype(bool))).astype(int)
    got = np.asarray(rp.code.scram.load(name).decode(bad.astype(bool))).astype(int)
    assert (ref != got).sum() == len(delays) + 1

def test_fdt_matches_wifi_dsss_descrambler(rng):
    """802.11b DSSS: self-synchronizing x^7 + x^4 + 1, as projects/wifi/wifi_dsss.py descrambles"""
    from rpps.code.scram import serial
    s = serial.init("wifi_dsss", {"base": "scramble", "type": "feedthrough", "register": "fibonacci", "poly": [7, 4], "seed": [0]*7})
    b = bits(rng, 500).astype(np.uint8)
    want = np.zeros(len(b), np.uint8)
    for n in range(7, len(b)):
        want[n] = b[n] ^ b[n - 4] ^ b[n - 7]
    got = np.asarray(s.decode(b.astype(bool))).astype(np.uint8)
    assert np.array_equal(got[7:], want[7:])

def test_fdt_streams_continue_between_calls(rng):
    s = rp.code.scram.load("v35")
    msg = bits(rng, 1000)
    whole = rp.code.scram.load("v35").encode(msg)
    parts = np.concatenate((s.encode(msg[:300]), s.encode(msg[300:])))
    assert np.array_equal(whole, parts)

def test_ccsds_sequence():
    """CCSDS 131.0-B pseudo-random sequence: s[n] = s[n-1]^s[n-3]^s[n-5]^s[n-8] from all ones, FF 48 0E C0 9A 0D 70 BC..."""
    s = [1]*8
    for n in range(8, 255*2):
        s.append(s[n-1] ^ s[n-3] ^ s[n-5] ^ s[n-8])
    y = np.asarray(rp.code.scram.load("ccsds").encode(np.zeros(len(s), dtype=bool))).astype(int)
    assert list(y) == s
    assert bytes(np.packbits(y.astype(np.uint8)))[:8].hex() == "ff480ec09a0d70bc"
