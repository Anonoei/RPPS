import numpy as np
import pytest

import rpps as rp
from conftest import bits

ALL = [(k, n) for k, names in rp.scram.ls().items() for n in names]

def _load(folder, name):
    """With example values for the parameters a definition needs (a cell id, a WACN ...)"""
    return rp._config.load("scram", name, folder, example=True)

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
    s = rp.scram.load("fdt", name)
    taps = [int(t) for t in s.s_lfsr._poly] if hasattr(s.s_lfsr, "_poly") else [int(t) for t in s.s_lfsr.taps]
    N = max(taps)
    y = np.asarray(s.encode(np.zeros(2000, dtype=bool))).astype(int)
    assert 0.45 < y.mean() < 0.55
    for i in range(N, len(y)):
        assert y[i-N] == y[i] ^ np.bitwise_xor.reduce([y[i-t] for t in taps if t < N])

def test_fdt_streams_continue_between_calls(rng):
    s = rp.scram.load("fdt", "v35")
    msg = bits(rng, 1000)
    whole = rp.scram.load("fdt", "v35").encode(msg)
    parts = np.concatenate((s.encode(msg[:300]), s.encode(msg[300:])))
    assert np.array_equal(whole, parts)

def test_ccsds_sequence():
    """CCSDS 131.0-B pseudo-random sequence: s[n] = s[n-1]^s[n-3]^s[n-5]^s[n-8] from all ones, FF 48 0E C0 9A 0D 70 BC..."""
    s = [1]*8
    for n in range(8, 255*2):
        s.append(s[n-1] ^ s[n-3] ^ s[n-5] ^ s[n-8])
    y = np.asarray(rp.scram.load("ccsds").encode(np.zeros(len(s), dtype=bool))).astype(int)
    assert list(y) == s
    assert bytes(np.packbits(y.astype(np.uint8)))[:8].hex() == "ff480ec09a0d70bc"
