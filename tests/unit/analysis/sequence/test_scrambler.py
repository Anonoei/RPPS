"""Blind scrambler recovery"""
import numpy as np

import rpps as rp
from rpps.analysis.sequence import scrambler as SC
from support.blind import _feedthrough, scramble

def test_descramble_is_the_feedthrough_definition():
    y = np.random.default_rng(0).integers(0, 2, 2000).astype(bool)
    d = np.asarray((rp.Data(y, flush=True) / rp.code.scram.load("feedthrough/v35")).data).astype(np.uint8)
    assert np.array_equal(rp.code.scram.Feedthrough.descramble(y, (20, 3))[20:], d[20:])

def test_scrambler_found_from_the_structure_under_it(hdlc):
    h, n, g = SC.search(scramble(hdlc, (20, 3)))
    assert h[0].params["taps"] == [20, 3] and g.definition["poly"] == [20, 3]
    h, n, g = SC.search(scramble(np.ones(60000, np.uint8), (17, 12))) # An idle line of ones: the strongest structure
    assert g.definition["poly"] == [17, 12]

def test_three_taps(hdlc):
    assert not SC.search(scramble(hdlc, (23, 18, 5)))[0]
    assert SC.search(scramble(hdlc, (23, 18, 5)), max_taps=3)[2].definition["poly"] == [23, 18, 5]

def test_white_stays_white(hdlc):
    rng = np.random.default_rng(2)
    for y in (rng.integers(0, 2, 100000).astype(np.uint8), scramble(rng.integers(0, 2, 60000).astype(np.uint8), (20, 3))):
        h, n, g = SC.search(y)
        assert not h and g is None
    h, n, g = SC.search(hdlc) # Structure with no scrambler: "none" stands out, nothing to save
    assert h[0].params["taps"] == [] and g is None

def _lfsr(taps, n, rng):
    s = list(rng.integers(0, 2, max(taps)))
    while len(s) < n:
        s.append(int(np.bitwise_xor.reduce([s[-t] for t in taps])))
    return np.array(s[:n], np.uint8)

def test_berlekamp_massey_finds_the_register():
    from rpps.math import gf2
    rng = np.random.default_rng(0)
    C, L = gf2.berlekamp_massey(_lfsr((23, 18), 400, rng))
    assert (L, gf2.lfsr_taps(C, L)) == (23, (23, 18))
    _, L = gf2.berlekamp_massey(rng.integers(0, 2, 400).astype(np.uint8))
    assert 190 <= L <= 210 # Random bits: about half their length

def test_a_prbs_and_an_additive_scrambler_over_idle_data():
    from rpps.analysis.sequence import scrambler as SC
    rng = np.random.default_rng(1)
    h, _, g = SC.additive_search(_lfsr((23, 18), 40000, rng))
    assert h[0].params["taps"] == [23, 18] and h[0].params["prbs"] and "PRBS23" in h[0].name
    d = rng.integers(0, 2, 40000).astype(np.uint8)
    for a in range(0, 40000, 4000):
        d[a:a + 1500] = 0 # Idle stretches: the scrambler's own sequence shows through
    h, _, g = SC.additive_search(d ^ _lfsr((15, 14), 40000, rng))
    assert h[0].params["taps"] == [15, 14] and not h[0].params["prbs"] and g is not None
    assert SC.additive_search(rng.integers(0, 2, 40000).astype(np.uint8))[0] == []

def test_cascaded_scramblers_over_a_repeating_word_are_explained():
    # Two scramblers (1+x^18+x^23) in cascade over a 24-bit word sent again and again
    from rpps.analysis.sequence import scrambler as SC
    word = np.random.default_rng(24).integers(0, 2, 24).astype(np.uint8)
    y = _feedthrough(_feedthrough(np.tile(word, 1700), (23, 18)), (23, 18))
    h, _, _ = SC.additive_search(y)
    p = h[0].params
    assert p["data_period"] == 24 and p["scramblers"][0]["taps"] == [23, 18] and p["scramblers"][0]["times"] == 2
    assert p["pattern_holds"] > 0.99 and p["pattern"] in "".join(map(str, np.tile(word, 3)))
    assert SC.explain([23, 18]) [1] == 1 # A scrambler over constant data: period 1
