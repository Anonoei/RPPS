"""Turbo codes and rate matching"""
import numpy as np
import pytest

import rpps as rp
from support.signals import bits

# --- turbo, rate matching -------------------------------------------------------------------------------------

def test_turbo_tail_terminates(rng):
    c = rp.code.types.rsc("0o13", "0o15", 4)
    x = rng.integers(0, 2, 101).astype(np.uint8)
    _, tx, _ = c.encode(x)
    s = 0
    for b in list(x) + list(tx):
        s = c.next[s, b]
    assert s == 0

def test_turbo_through_rate_matching(rng):
    K, E = 248, 900
    t = rp.code.load("block/turbo/lte", K=K).encoder
    good = 0
    for rv in range(4):
        rm = rp.code.load("block/ratematch/lte_turbo", K=K, E=E, rv=rv).encoder
        for _ in range(3):
            c = bits(rng, K)
            tx = rm.encode(t.encode(c))
            llr = 2*((1 - 2.0*tx) + rng.normal(0, 0.9, E))/0.81
            good += np.array_equal(t.decode(rm.decode(llr)), c)
    assert good >= 11

def test_turbo_qpp_needs_a_listed_size():
    with pytest.raises(ValueError):
        rp.code.load("block/turbo/lte", K=41)

def test_conv_rate_matching_repeats(rng):
    D, E = 40, 1920
    rm = rp.code.load("block/ratematch/lte_conv", D=D, E=E).encoder
    x = bits(rng, 3*D)
    y = rm.encode(x)
    assert len(y) == E
    sums = rm.decode(1.0 - 2.0*y)
    assert np.allclose(np.abs(sums), E/(3*D)) # Each coded bit repeated 16 times
    assert np.array_equal(sums < 0, x)

def test_turbo_rate_matching_punctures(rng):
    K = 40
    rm = rp.code.load("block/ratematch/lte_turbo", K=K, E=100, rv=0).encoder
    sums = rm.decode(np.ones(100))
    assert (sums > 0).sum() == 100 and len(sums) == 3*(K + 4) # 100 distinct positions, the rest erased

@pytest.mark.parametrize("rate, n", [("1_2", 2), ("1_3", 3), ("1_4", 4), ("1_6", 6)])
def test_ccsds_turbo_lengths_and_decoding(rate, n, rng):
    """CCSDS 131.0-B-5 6: (k + 4)/r symbols a word; decodes at Eb/N0 1.5 dB"""
    c = rp.code.load(f"block/turbo/ccsds_{rate}", k=1784)
    assert c.encoder.den == n*(1784 + 4) and c.encoder.num == 1784
    msg = rng.integers(0, 2, 1784).astype(bool)
    x = c.encoder.encode(msg)
    assert np.array_equal(x[:2*n:n], msg[:2]) # the systematic bit leads every bit time
    ebn0 = 10**(1.5/10)
    sigma = np.sqrt(n/(2*ebn0))
    llr = 2*((1 - 2.0*x) + rng.normal(0, sigma, len(x)))/sigma**2
    assert np.array_equal(c.decoder.decode(llr), msg)

def test_ccsds_turbo_terminates_both_encoders(rng):
    c = rp.code.load("block/turbo/ccsds_1_3", k=1784).encoder
    sa, _ = c._run(rng.integers(0, 2, 1784).astype(np.uint8))
    assert len(sa) == 1788
