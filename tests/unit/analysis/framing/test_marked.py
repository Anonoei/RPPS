"""Frames behind a sync marker (rpps.analysis.framing.marked): the marker's period, the scrambler restarted at each, the
Reed-Solomon code and its interleaving depth inside; and identify naming the CCSDS TM profiles from them"""
import numpy as np
import pytest

import rpps as rp
from rpps.analysis import identify
from rpps.analysis.framing import marked as MK

def channel_bits(name, bits, depth):
    """Bits through a profile's code steps: the channel bits its receiver decodes"""
    d = rp.Data(bits, Fs=1e6, sps=1, flush=True)
    for how, op in rp.profile.load(name, depth=depth).tx():
        if how == "*" and op.CATEGORY == rp.Category.CODE:
            d = d*op
    return np.asarray(d.data).astype(np.uint8)

def test_markers_are_the_profiles_own():
    m = MK.markers()
    assert "ccsds_asm" in m and len(m["ccsds_asm"]) == 32 and all(len(b) >= MK.MIN_MARKER for b in m.values())

def test_chance_of_a_random_word_decoding():
    """RS(255,223) corrects 16: a random word lands within 16 of a codeword about 3e-14 of the time"""
    from rpps.code.coding import cached
    assert 1e-15 < MK.chance(cached("block/rs/ccsds_255_223_dual").encoder) < 1e-13

@pytest.mark.parametrize("depth, ber", [(1, 0.0), (5, 0.0), (5, 5e-3)])
def test_layer_finds_the_frames_scrambler_and_code(depth, ber, rng):
    """CCSDS TM channel bits after a random lead, with bit errors: 5e-3 leaves a few errors a word, which the code
    corrects, so the words still decode (15 or 16 of 16 over seeds)"""
    x = channel_bits("ccsds/tm_rs", rng.integers(0, 2, 8*223*depth*10).astype(bool), depth)
    x = np.concatenate((rng.integers(0, 2, 777).astype(np.uint8), x))
    x ^= (rng.random(len(x)) < ber).astype(np.uint8)
    f = MK.layer(x)
    assert (f.marker, f.period, f.first, f.inverted) == ("ccsds_asm", 32 + 8*255*depth, 777, False)
    assert (f.code, f.interleaver, f.depth) == ("rs/ccsds_255_223_dual", "block/ccsds_rs", depth)
    assert f.ok >= f.words - 2 and f.z > 20
    if depth > 1: # (at depth 1 the randomizer's 255-bit period makes its keystream a codeword: either decodes)
        assert f.scrambler == "additive/ccsds"
    assert MK.layer(x ^ 1).inverted

def test_layer_on_random_bits_is_nothing(rng):
    f = MK.layer(rng.integers(0, 2, 200000).astype(np.uint8))
    assert f is None or f.code is None

@pytest.mark.parametrize("name", ["ccsds/tm_rs", "ccsds/tm_concat"])
@pytest.mark.parametrize("depth", [1, 5])
def test_identify_names_the_ccsds_tm_profiles(name, depth):
    """BPSK at about 9 dB, its phase turned: the marker-framed RS code found on the channel bits (tm_rs) or behind the
    Viterbi (tm_concat), the profile named; tm_packets reads the same code steps and extends tm_rs, so tm_rs is named"""
    rng = np.random.default_rng(depth)
    x = channel_bits(name, rng.integers(0, 2, 8*223*depth*10).astype(bool), depth).astype(bool)
    s = np.asarray((rp.Data(x) * rp.symbol.load("bpsk")).data)
    s = (s*np.exp(0.3j) + 0.25*(rng.normal(size=len(s)) + 1j*rng.normal(size=len(s)))).astype(np.complex64)
    d = identify.identify(s, "bpsk", blind=False)
    assert d.found and d.profile_name == name
