"""Every shipped FEC definition loads, and encode -> decode gives the message back"""
import numpy as np
import pytest

import rpps as rp
from conftest import bits

BIG = ("dvbs2_",) # DVB-S2 frames (16200+ bits) are tested separately: too slow to sweep

def configs(kind):
    return [n for n in rp.code.names(kind) if not any(b in n for b in BIG)]

def load(kind, name, **kw):
    """With the example values of any parameters the definition needs (a block size ...)"""
    ex = rp._config.examples(rp._config.read(rp.code.find(kind, name)))
    return rp.code.load(kind, name, **ex, **kw)

def test_kinds():
    assert set(rp.code.ls()) >= {"block/rs", "conv/k7_171_133", "interleave/forney"}

def test_unknown_definition():
    with pytest.raises(FileNotFoundError):
        rp.code.load("block", "does.not.exist")

@pytest.mark.parametrize("name", configs("block"))
def test_block_round_trip(name, rng):
    code = load("block", name)._enc
    if name.startswith("rs/"):
        msg = bits(rng, code.num*code.m*5)
        per = code.m # Lengths count symbols of m bits
    else:
        msg = bits(rng, code.num*20)
        per = 1
    coded = code.encode(msg)
    tail = getattr(code, "tail", 0)*getattr(code, "n", 0) # A trellis code's flush steps
    assert len(coded) == len(msg)//(code.num*per)*code.den*per + tail
    if getattr(code, "levels", None) is not None: # Decoded from the amplitudes it sends
        coded = code.modulate(coded)
    out = np.asarray(code.decode(coded))
    if type(code).__name__ == "ratematch": # Rate matching gives back summed soft values
        out = out < 0
    assert np.array_equal(out.astype(bool), msg)

@pytest.mark.parametrize("name", configs("block"))
def test_block_rate_matches_properties(name):
    c = load("block", name)
    assert c.e_rate == pytest.approx(c.e_num/c.e_den)
    assert 0 < c.e_rate <= 1

def test_linear_syndrome_decoder_corrects_one_error_per_block():
    c = rp.code.load("block", "hamming/7_4", decoder="linear")._dec
    msg = np.array([1, 0, 1, 1], dtype=bool)
    coded = c.encode(msg).copy()
    coded[2] ^= True
    assert np.array_equal(c.decode(coded), msg)

def test_rs_corrects_errors(rng):
    rs = rp.code.load("block", "rs/204_188")._enc
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
    code = rp.code.load("conv", name)
    msg = bits(rng, 2400)
    out = np.asarray(code._dec.decode(code._enc.encode(msg))).astype(bool)
    n = min(len(out), len(msg)) - 40 # The tail of a non-terminated stream is not guaranteed
    assert n > 2000
    assert np.array_equal(out[:n], msg[:n])

def test_viterbi_soft_beats_hard(rng):
    code = rp.code.load("conv", "k7_133_171/1_2")
    msg = bits(rng, 4000)
    coded = code._enc.encode(msg)
    sigma = 0.75
    rx = (1.0 - 2.0*coded.astype(float)) + rng.normal(0, sigma, len(coded))
    hard = np.asarray(code._dec.decode((rx < 0).astype(np.uint8))).astype(bool)
    soft = np.asarray(code._dec.decode(rx, soft=True)).astype(bool)
    n = min(len(hard), len(soft), len(msg)) - 40
    assert (soft[:n] != msg[:n]).sum() < (hard[:n] != msg[:n]).sum()

def test_interleaver_round_trip(rng):
    il = rp.code.load("interleave", "forney/12_17")._enc
    x = rng.integers(0, 256, 204*30).astype(np.uint8)
    y = il.decode(il.encode(x, symbols=True), symbols=True)
    delay = (il.I - 1)*il.M*il.I # Total latency of interleaver + deinterleaver
    assert np.array_equal(np.asarray(y)[delay:delay+1000], x[:1000])

def test_ldpc_corrects_noise(rng):
    code = rp.code.load("block", "ldpc/gallager_96_48")._enc
    msg = bits(rng, code.num*40)
    coded = code.encode(msg)
    rx = (1.0 - 2.0*coded.astype(float)) + rng.normal(0, 0.6, len(coded))
    llr = 2*rx/0.6**2
    out = np.asarray(code.decode(llr)).astype(bool)
    assert (out != msg).mean() < 0.01
    assert code.converged.mean() > 0.9

def test_registry_lists_all_types():
    assert {"linear", "repeat", "rs", "bch", "ldpc", "conv", "viterbi", "forney"} <= set(rp.code.types.REGISTRY)

# MacKay alist of the Hamming (7,4) parity checks: 7 bits, 3 checks
HAMMING_ALIST = """7 3
3 4
1 1 2 2 1 2 3
4 4 4
1 0 0
2 0 0
1 2 0
1 3 0
3 0 0
2 3 0
1 2 3
1 3 4 7
2 3 6 7
4 5 6 7
"""

def test_import_alist(tmp_path):
    f = tmp_path / "ham.alist"
    f.write_text(HAMMING_ALIST)
    obj = rp.code.import_alist(f)
    assert obj["n"] == 7 and obj["checks"] == [[0, 2, 3, 6], [1, 2, 5, 6], [3, 4, 5, 6]]
    path = rp.code.import_alist(f, "alist_ham", dest=tmp_path / "cfg", comment="test")
    rp.add_config_path(tmp_path / "cfg")
    try:
        code = rp.code.load("block", "ldpc/alist_ham")._enc
        assert (code.num, code.den) == (4, 7)
        word = code.encode(np.array([1, 0, 1, 1], dtype=bool))
        assert not np.any(code.syndrome(word[None, :]))
        bad = word.copy()
        bad[2] ^= True
        assert np.any(code.syndrome(bad[None, :]))
    finally:
        rp._config._EXTRA.remove(tmp_path / "cfg")

def test_import_alist_rejects_inconsistent(tmp_path):
    f = tmp_path / "bad.alist"
    f.write_text(HAMMING_ALIST.replace("2 3 6 7", "2 3 5 7"))
    with pytest.raises(ValueError, match="different matrices"):
        rp.code.import_alist(f)
