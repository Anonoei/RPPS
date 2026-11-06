import numpy as np
import pytest

import rpps as rp
from conftest import bits

ALL = [(k, n) for k, names in rp.scram.ls().items() for n in names]

@pytest.mark.parametrize("folder,name", ALL)
def test_scram_round_trip(folder, name, rng):
    s = rp.scram.load(folder, name)
    msg = bits(rng, 8*204*8) # Whole frames for the frame-based scramblers
    scr = s.encode(msg)
    assert not np.array_equal(scr, msg)
    assert np.array_equal(np.asarray(s.decode(scr)).astype(bool), msg)

@pytest.mark.parametrize("folder,name", ALL)
def test_scram_whitens_constant_input(folder, name):
    s = rp.scram.load(folder, name)
    scr = np.asarray(s.encode(np.zeros(8*204*8, dtype=bool)))
    assert 0.4 < scr.mean() < 0.6

@pytest.mark.parametrize("name", ["v35", "galois", "fib"])
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
