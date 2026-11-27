"""Block codes defined bit by bit: parity equations, expression codes, appended check bits, layout and xor
(rpps.code.types.layout, linear "parity", type "expr")"""
import numpy as np
import pytest

import rpps as rp
from rpps.code import serial
from rpps.code.types.layout import arranged

H74 = {"base": "block", "type": "linear", "n": 7, "k": 4, "parity": [[0, 1, 3], [0, 2, 3], [1, 2, 3]]}

def _roundtrip(code, rng, flips, soft, words=40):
    k = code.encoder.num
    u = rng.integers(0, 2, k*words).astype(bool)
    x = np.asarray(code.encoder.encode(u)).astype(bool).reshape(words, -1).copy()
    for w in x:
        w[rng.choice(len(w), flips, replace=False)] ^= True
    x = x.reshape(-1)
    y, errs = code.decoder.decode(np.where(x, -4.0, 4.0) if soft else x, status=True)
    return np.asarray(y).astype(bool), u, np.asarray(errs)

def test_parity_equations_are_the_p_block():
    p = [[1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]] # message bit i's row: the parity bits it feeds
    a = serial.init("a", H74)
    b = serial.init("b", {"base": "block", "type": "linear", "n": 7, "k": 4, "p": p})
    assert np.array_equal(a.encoder.generator, b.encoder.generator)

def test_two_matrix_forms_are_an_error():
    with pytest.raises(ValueError, match="one of"):
        serial.init("x", {**H74, "p": [[1, 1, 0]]*4})

@pytest.mark.parametrize("soft", [False, True])
@pytest.mark.parametrize("extra", [{}, {"layout": {"message": [2, 4, 5, 6], "parity": "rest"}},
                                   {"layout": [6, 5, 4, 3, 2, 1, 0], "xor": "0x55"},
                                   {"layout": [7, 6, 5, 4, 3, 2, 1, 0], "xor": [1, 0, 0, 1, 1, 0, 1, 0], "extend": True}])
def test_layout_xor_extend_correct_one_error(rng, soft, extra):
    c = serial.init("h", {**H74, **extra})
    y, u, errs = _roundtrip(c, rng, 1, soft)
    assert np.array_equal(y, u) and (errs == 1).all()

def test_layout_puts_bits_where_it_says():
    c = serial.init("h", {**H74, "layout": {"message": [6, 5, 4, 3], "parity": "rest"}})
    u = np.array([1, 0, 1, 1], bool)
    plain = serial.init("p", H74).encoder.encode(u)
    sent = c.encoder.encode(u)
    assert list(sent[[6, 5, 4, 3]]) == list(u) and list(sent[[0, 1, 2]]) == list(plain[4:])

def test_layout_must_place_every_bit():
    with pytest.raises(ValueError, match="placement"):
        serial.init("h", {**H74, "layout": [0, 1, 2, 3, 4, 5, 5]})

def test_expr_affine_loads_linear_with_its_arrangement(rng):
    # The message bits scattered, a parity bit inverted: the same code as H74 sent in another order, offset
    bits = ["m0 ^ m1 ^ m3", "m0 ^ m2 ^ m3 ^ 1", "m0", "m1 ^ m2 ^ m3", "m1", "m2", "m3"]
    c = serial.init("e", {"base": "block", "type": "expr", "k": 4, "bits": bits})
    assert isinstance(c.encoder, arranged) and type(c.decoder.dec).__name__ == "trellis"
    u = np.array([1, 0, 1, 1], bool)
    assert c.encoder.encode(u).astype(int).tolist() == [0, 0, 1, 0, 0, 1, 1]
    for soft in (False, True):
        y, u, errs = _roundtrip(c, rng, 1, soft)
        assert np.array_equal(y, u) and (errs == 1).all()

def test_expr_nonlinear_decodes_from_its_codebook(rng):
    bits = ["m0", "m1", "m2", "m0 & m1", "m1 | m2", "m0 ^ m2", "~m1", "m0 ^ m1 ^ m2"]
    c = serial.init("n", {"base": "block", "type": "expr", "k": 3, "bits": bits})
    assert type(c.decoder).__name__ == "expr"
    for soft in (False, True):
        y, u, errs = _roundtrip(c, rng, 1, soft)
        assert np.array_equal(y, u) and (errs == 1).all()

def test_expr_refuses_a_big_codebook():
    with pytest.raises(ValueError, match="k=17"):
        serial.init("n", {"base": "block", "type": "expr", "k": 17, "bits": [f"m{i}" for i in range(17)] + ["m0 & m1"]})

P25_NID_P = """6331 1413 6723 5452|5265 5216 1472 3276|4603 7114 6116 4164|2301 7446 3047 2072|7271 6230 7300 0466|
5605 6507 5263 5660|2702 7243 6531 6730|1341 3521 7254 7354|0560 5650 7526 3566|6141 3337 5170 4220|3060 5557 6474 2110|
1430 2667 7236 1044|0614 1333 7517 0422|6037 1146 1164 1642|5326 5070 6351 5373|4662 3027 5647 3127"""

def test_p25_nid_is_the_spec_generator(rng):
    """TIA-102.BAAA-A 8.5.2: the 16 x 64 generator matrix, BCH (63,16) then P = S1 xor S0 (append [[14, 15]])"""
    P = np.array([[int(b) for b in format(int(r.strip().replace(" ", ""), 8), "048b")] for r in P25_NID_P.split("|")])
    G = np.concatenate((np.eye(16, dtype=int), P), axis=1)
    c = rp.code.load("block/bch/p25_nid")
    u = rng.integers(0, 2, (50, 16))
    assert np.array_equal(c.encoder.encode(u.reshape(-1).astype(bool)).reshape(50, -1).astype(int), (u @ G) % 2)
    y, u2, errs = _roundtrip(c, rng, 11, False)
    assert np.array_equal(y, u2) and (errs == 11).all()
    y, u2, errs = _roundtrip(c, rng, 13, False)
    assert (errs[~np.all(y.reshape(-1, 16) == u2.reshape(-1, 16), axis=1)] == -1).all() # a wrong word is never passed as decoded

def test_append_checks_bits_against_the_decision(rng):
    c = serial.init("b", {"base": "block", "type": "bch", "n": 63, "k": 16, "m": 6, "poly": "0x43", "t": 11, "extend": True})
    assert c.encoder.den == 64 and not (c.encoder.encode(np.ones(16, bool)).sum() % 2)
    y, u, errs = _roundtrip(c, rng, 11, False)
    assert np.array_equal(y, u)
    _, _, errs = _roundtrip(c, rng, 12, False)
    assert (errs == -1).all()

def test_layout_in_a_block_step(rng):
    """The arranged code works as a profile's block step does: through the operator, any cut"""
    c = serial.init("h", {**H74, "layout": [6, 5, 4, 3, 2, 1, 0], "xor": "0x0f"})
    u = rng.integers(0, 2, 4*30).astype(bool)
    tx = rp.Data(u) * c
    rx = rp.Data(np.asarray(tx.data)) / c
    assert np.array_equal(np.asarray(rx.data).astype(bool), u)

def test_a_decoder_a_code_cannot_have_is_a_clear_error():
    bits = ["m0", "m1", "m2", "m0 & m1"] # not linear: no matrices for a trellis
    with pytest.raises(rp.errors.ConfigError, match="leave out"):
        serial.init("n", {"base": "block", "type": "expr", "k": 3, "bits": bits, "decoder": "trellis"})
    with pytest.raises(rp.errors.ConfigError, match="not a decoder"):
        serial.init("b", {"base": "block", "type": "bch", "n": 63, "k": 16, "m": 6, "poly": "0x43", "t": 11, "decoder": "viterbi"})
    c = serial.init("e", {"base": "block", "type": "expr", "k": 3, "bits": ["m0", "m1", "m2", "m0 ^ m1"], "decoder": "syndrome"})
    assert type(getattr(c.decoder, "dec", c.decoder)).__name__ == "syndrome" # (an affine expr code is linear: any matrix decoder)
