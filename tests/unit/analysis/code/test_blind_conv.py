"""Blind convolutional codes and puncture patterns"""
import numpy as np
import pytest

import rpps as rp
from rpps.analysis import identify as F
from rpps.logger import Logger
from rpps.analysis.code.conv import puncture as PU
from rpps.code.types.conv import conv
from support.blind import _qpsk

@pytest.mark.kernels
def test_unknown_puncture_pattern_on_a_known_mother_code():
    rng = np.random.default_rng(2)
    g = [[1, 0, 1, 1, 0, 1, 1], [1, 1, 1, 1, 0, 0, 1]]
    pat = [[1, 0, 1, 0, 1, 1], [1, 1, 0, 1, 0, 0]] # Rate 6/7, in no definition
    bits = conv(Logger().Child("t", 0), 1, 2, g, 7, pat).encode(rng.integers(0, 2, 20000).astype(bool))[3:]
    q = rp.symbol.load("qpsk")
    s = q.modulate(bits[:len(bits)//2*2])*1j
    s = s + 0.15*(rng.standard_normal(len(s)) + 1j*rng.standard_normal(len(s)))
    v, guess = PU.best(*PU.search(F.bit_streams(s[:8000], q, differential=False)))
    assert guess.definition["extends"] == rp.code.serial.mother(g, 7) == "k7_133_171" # a child of the mother code it found
    found = np.array(guess.definition["puncture"], dtype=bool)
    assert any(np.array_equal(found, np.roll(np.array(pat, dtype=bool), sh, axis=1)) for sh in range(6)) # (up to a turn)
    assert v.params["mismatch"] < 0.01
    q2 = q.points[rng.integers(0, 4, 8000)] + 0.1*(rng.standard_normal(8000) + 1j*rng.standard_normal(8000))
    assert PU.best(*PU.search(F.bit_streams(q2, q, differential=False))) == (None, None)

@pytest.mark.parametrize("P", [2, 3, 5])
def test_puncture_mismatches_in_one_batch_are_each_patterns_own(P):
    """Every pattern of a mother decoded together on the mother's trellis: the counts mismatch_batch gives under each
    pattern's own punctured code, row for row (a codeword stream, a punctured one at every offset, random bits)"""
    from rpps.code.coding import mismatch_batch
    rng = np.random.default_rng(P)
    K, gen = 7, ((1, 0, 1, 1, 0, 1, 1), (1, 1, 1, 1, 0, 0, 1))
    pats = PU.patterns(P)
    sent = conv(Logger().Child("t", 0), 1, 2, [list(g) for g in gen], 7, pats[0].astype(int).tolist()).encode(
        rng.integers(0, 2, 3000).astype(bool)).astype(np.uint8)
    rows = []
    for pat in pats:
        m, per = int(pat.sum()), 400//P
        rows += [(pat, sent[off:off + per*m]) for off in range(m)]
        rows += [(pat, rng.integers(0, 2, per*m).astype(np.uint8)), (pat, rng.integers(0, 2, 4*per*m).astype(np.uint8))]
    k, n = PU.mismatches(K, gen, rows)
    for (pat, h), kk, nn in zip(rows, k, n):
        want, wn = mismatch_batch(PU._code(K, gen, pat), h[None])
        assert (kk, nn) == (int(want[0]), wn)
    assert min(k[:int(pats[0].sum())]) == 0 # (the stream's own pattern, at its offset: nothing differs)

def test_an_inverted_output_is_the_code_with_that_output_flipped():
    # CCSDS 131.0-B 3.3: the basic K=7 code sends its G2 output inverted
    rng = np.random.default_rng(4)
    m = rng.integers(0, 2, 500).astype(bool)
    plain = rp.code.load("conv/k7_171_133/1_2").encoder.encode(m)
    ccsds = rp.code.load("conv/k7_171_133/ccsds_1_2")
    e = ccsds.encoder.encode(m)
    assert np.array_equal(e[0::2], plain[0::2]) and np.array_equal(e[1::2], ~plain[1::2])
    assert np.array_equal(ccsds.decoder.decode(e.astype(np.uint8))[:len(m)], m)

@pytest.mark.parametrize("name", ["k7_133_171/2_3", "k7_133_171/5_6", "k7_133_171/7_8", "k9_561_753/1_2",
                                  "k9_557_663_711/1_3", "k7_133_171_165/1_3", "k7_133_171_145_133/1_4"])
def test_standard_convolutional_codes_round_trip(name):
    rng = np.random.default_rng(5)
    c = rp.code.load(f"conv/{name}")
    m = rng.integers(0, 2, 1260).astype(bool) # A whole number of every puncture period (2, 3, 5, 7)
    e = c.encoder.encode(m)
    assert np.array_equal(c.decoder.decode(e.astype(np.uint8))[:len(m)], m)

def test_a_ccsds_stream_on_bpsk_is_identified():
    # On BPSK the G2 inversion turns every other symbol: no map of the plain code fits, the CCSDS definition does
    rng = np.random.default_rng(6)
    e = rp.code.load("conv/k7_171_133/ccsds_1_2").encoder.encode(rng.integers(0, 2, 20000).astype(bool))
    y = (1 - 2*e.astype(float)) + 0.3*rng.standard_normal(len(e))
    out = rp.analysis.identify.identify(y.astype(np.complex64), "bpsk")
    assert out.found and out.name == "k7_171_133/ccsds_1_2"

@pytest.mark.kernels
def test_an_unknown_convolutional_code_is_not_taken_for_a_block_code():
    # K=6 (65, 57), in no definition: a convolutional code is linear over any block length, so the block search would
    # name it a code of some length; the convolutional scan goes first and gives its generators
    rng = np.random.default_rng(6)
    g = [[1, 1, 0, 1, 0, 1], [1, 0, 1, 1, 1, 1]]
    s = _qpsk(conv(Logger().Child("t", 0), 1, 2, g, 6, [[1], [1]]).encode(rng.integers(0, 2, 40000).astype(bool))[1:], rng)
    f = F.identify(s[:2**14], "qpsk", blind=True)
    assert [h.kind for h in f.inner][:1] == ["conv"]
    assert sorted(map(list, f.guesses[0].definition["generator"])) == sorted(g)
