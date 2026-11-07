"""Codes built from parameters match the ones loaded from definitions"""
import numpy as np
import pytest

import rpps as rp

def same_code(a, b, rng, k):
    msg = rng.integers(0, 2, k*8).astype(bool)
    ea, eb = (rp.Data(msg, flush=True) * c for c in (a, b))
    assert np.array_equal(ea.data, eb.data)
    return msg, ea

@pytest.mark.parametrize("n,k", [(3, 1), (7, 4), (15, 11)])
def test_hamming_matches_the_definitions(n, k, rng):
    ours, theirs = rp.coding.Hamming(n, k), rp.coding.load("block", f"hamming/{n}_{k}")
    msg = rng.integers(0, 2, k*8).astype(bool)
    a, b = (c._enc.encode(msg) for c in (ours, theirs))
    # Same code up to the order of the parity columns: both are single error correcting with k data bits first
    assert np.array_equal(a.reshape(-1, n)[:, :k], b.reshape(-1, n)[:, :k])
    cw = a.copy().reshape(-1, n)
    cw[:, n//2] ^= True
    assert np.array_equal(np.asarray(ours.decode(rp.Data(cw.reshape(-1), flush=True)).data), msg)

def test_hamming_rejects_what_is_not_a_code():
    with pytest.raises(ValueError):
        rp.coding.Hamming(7, 5)

def test_shortened_hamming_corrects_a_single_error(rng):
    code = rp.coding.Hamming(12, 8)
    msg = rng.integers(0, 2, 8*20).astype(bool)
    cw = (rp.Data(msg, flush=True) * code).data.reshape(-1, 12).copy()
    cw[:, 5] ^= True
    assert np.array_equal((rp.Data(cw.reshape(-1), flush=True) / code).data, msg)

def test_repetition(rng):
    code = rp.coding.Repetition(5)
    msg = rng.integers(0, 2, 100).astype(bool)
    cw = (rp.Data(msg, flush=True) * code).data
    assert len(cw) == 500
    cw = cw.copy()
    cw[::5] ^= True; cw[1::5] ^= True # Two of every five wrong
    assert np.array_equal((rp.Data(cw, flush=True) / code).data, msg)

def test_rs_matches_the_definition_and_corrects(rng):
    ours, theirs = rp.coding.RS(204, 188), rp.coding.load("block", "rs/204_188")
    msg = rng.integers(0, 256, 188*3).astype(np.uint8)
    a, b = (c._enc.encode(msg, symbols=True) for c in (ours, theirs))
    assert np.array_equal(a, b)
    bad = a.copy().reshape(-1, 204)
    bad[:, [3, 99, 150]] ^= 0x5A
    assert np.array_equal(np.asarray(ours._enc.decode(bad.reshape(-1), symbols=True)).reshape(-1), msg)

def test_rs_255_223(rng):
    rs = rp.coding.RS(255, 223)._enc
    msg = rng.integers(0, 256, 223*2).astype(np.uint8)
    cw = rs.encode(msg, symbols=True).reshape(-1, 255).copy()
    cw[:, :16] ^= 0x33 # t = 16 errors
    assert np.array_equal(np.asarray(rs.decode(cw.reshape(-1), symbols=True)).reshape(-1), msg)

def test_conv_matches_the_definition(rng):
    ours = rp.coding.Conv([0o133, 0o171], 7)
    theirs = rp.coding.load("conv", "k7_133_171/1_2")
    msg = rng.integers(0, 2, 200).astype(bool)
    assert np.array_equal(ours._enc.encode(msg), theirs._enc.encode(msg))
    cw = ours._enc.encode(msg).copy()
    cw[[10, 80, 200]] ^= True
    assert np.array_equal(np.asarray(ours._dec.decode(cw))[:len(msg)-6], msg[:len(msg)-6])

def test_conv_with_a_puncture_pattern_and_bcjr(rng):
    code = rp.coding.Conv([[1, 0, 1, 1, 0, 1, 1], [1, 1, 1, 1, 0, 0, 1]], puncture=[[1, 1, 0], [1, 0, 1]], decoder="bcjr")
    ref = rp.coding.load("conv", "k7_133_171/3_4")
    msg = rng.integers(0, 2, 300).astype(bool)
    assert np.array_equal(code._enc.encode(msg), ref._enc.encode(msg))
    assert code.e_rate == pytest.approx(ref.e_rate)
