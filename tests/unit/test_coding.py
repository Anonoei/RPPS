"""Every shipped FEC definition loads, and encode -> decode gives the message back"""
import numpy as np
import pytest

import rpps as rp
from conftest import bits

BIG = ("dvbs2_",) # DVB-S2 frames (16200+ bits) are tested separately: too slow to sweep

def configs(kind):
    return [n for n in rp.coding.names(kind) if not any(b in n for b in BIG)]

def test_kinds_and_aliases():
    assert set(rp.coding.ls()) >= {"block/rs", "conv/dvbs", "interleave/forney"}
    assert rp.coding.names("blk") == rp.coding.names("block")
    assert rp.coding.names("cnv") == rp.coding.names("conv")

def test_unknown_definition():
    with pytest.raises(FileNotFoundError):
        rp.coding.load("block", "does.not.exist")

@pytest.mark.parametrize("name", configs("block"))
def test_block_round_trip(name, rng):
    code = rp.coding.load("block", name)._enc
    if name.startswith("rs."):
        msg = bits(rng, code.num*8*5)
    else:
        msg = bits(rng, code.num*20)
    coded = code.encode(msg)
    assert len(coded) == len(msg)//code.num*code.den
    assert np.array_equal(np.asarray(code.decode(coded)).astype(bool), msg)

@pytest.mark.parametrize("name", configs("block"))
def test_block_rate_matches_properties(name):
    c = rp.coding.load("block", name)
    assert c.e_rate == pytest.approx(c.e_num/c.e_den)
    assert 0 < c.e_rate <= 1

def test_linear_syndrome_decoder_corrects_one_error_per_block():
    c = rp.coding.load("block", "hamming.7_4", decoder="linear")._dec
    msg = np.array([1, 0, 1, 1], dtype=bool)
    coded = c.encode(msg).copy()
    coded[2] ^= True
    assert np.array_equal(c.decode(coded), msg)

def test_rs_corrects_errors(rng):
    rs = rp.coding.load("block", "rs.204_188")._enc
    msg = rng.integers(0, 256, 188*4).astype(np.uint8)
    coded = rs.encode(msg, symbols=True).reshape(-1, 204)
    bad = coded.copy()
    for w in range(len(bad)): # t = 8 symbol errors per word
        pos = rng.choice(204, 8, replace=False)
        bad[w, pos] ^= rng.integers(1, 256, 8).astype(np.uint8)
    out = rs.decode(bad.reshape(-1), symbols=True)
    assert np.array_equal(np.asarray(out).reshape(-1), msg)
    assert (rs.errors == 8).all()

@pytest.mark.parametrize("name", configs("conv"))
def test_conv_clean_round_trip(name, rng):
    code = rp.coding.load("conv", name)
    msg = bits(rng, 2400)
    out = np.asarray(code._dec.decode(code._enc.encode(msg))).astype(bool)
    n = min(len(out), len(msg)) - 40 # The tail of a non-terminated stream is not guaranteed
    assert n > 2000
    assert np.array_equal(out[:n], msg[:n])

def test_viterbi_soft_beats_hard(rng):
    code = rp.coding.load("conv", "iess308.1_2_7")
    msg = bits(rng, 4000)
    coded = code._enc.encode(msg)
    sigma = 0.75
    rx = (1.0 - 2.0*coded.astype(float)) + rng.normal(0, sigma, len(coded))
    hard = np.asarray(code._dec.decode((rx < 0).astype(np.uint8))).astype(bool)
    soft = np.asarray(code._dec.decode(rx, soft=True)).astype(bool)
    n = min(len(hard), len(soft), len(msg)) - 40
    assert (soft[:n] != msg[:n]).sum() < (hard[:n] != msg[:n]).sum()

def test_interleaver_round_trip(rng):
    il = rp.coding.load("interleave", "forney.12_17")._enc
    x = rng.integers(0, 256, 204*30).astype(np.uint8)
    y = il.decode(il.encode(x, symbols=True), symbols=True)
    delay = (il.I - 1)*il.M*il.I # Total latency of interleaver + deinterleaver
    assert np.array_equal(np.asarray(y)[delay:delay+1000], x[:1000])

def test_ldpc_corrects_noise(rng):
    code = rp.coding.load("block", "ldpc.gallager_96_48")._enc
    msg = bits(rng, code.num*40)
    coded = code.encode(msg)
    rx = (1.0 - 2.0*coded.astype(float)) + rng.normal(0, 0.6, len(coded))
    llr = 2*rx/0.6**2
    out = np.asarray(code.decode(llr)).astype(bool)
    assert (out != msg).mean() < 0.01
    assert code.converged.mean() > 0.9

def test_registry_lists_all_types():
    assert {"linear", "repeat", "rs", "bch", "ldpc", "conv", "viterbi", "forney"} <= set(rp.coding.types.REGISTRY)
