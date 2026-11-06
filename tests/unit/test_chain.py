import numpy as np
import pytest

import rpps as rp

pytestmark = pytest.mark.slow # The pure-Python Viterbi dominates: deselect with -m "not slow"

def ts_packets(rng, n):
    p = rng.integers(0, 256, (n, 188)).astype(np.uint8)
    p[:, 0] = 0x47
    return p

def test_dvbs_chain_round_trip(rng):
    chain = rp.Chain.load("dvbs_3_4")
    pk = ts_packets(rng, 8*16) # Whole 8-packet scrambler frames
    coded = chain.encode(pk)
    assert coded.dtype == bool
    res = chain.decode(coded)
    got = np.asarray(res.packets)
    assert res.stats["rs_ok"] > 0.9
    n = min(len(got), len(pk))
    assert n >= 8*8
    assert np.array_equal(got[:n][8:n-8], pk[:n][8:n-8]) # Ends: interleaver / Viterbi latency

def test_dvbs_chain_survives_inverted_bits(rng):
    chain = rp.Chain.load("dvbs_3_4")
    pk = ts_packets(rng, 8*16)
    coded = chain.encode(pk)
    res = chain.decode(~coded)
    assert res.stats["rs_ok"] > 0.9
