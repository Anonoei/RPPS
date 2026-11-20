"""Every shipped FEC definition loads, and encode -> decode gives the message back"""
from pathlib import Path

import numpy as np
import pytest

import rpps as rp
from conftest import bits

BIG = ("dvbs2_",) # DVB-S2 frames (16200+ bits) are tested separately: too slow to sweep

def configs(kind):
    return [n for n in rp.code.names(kind) if not any(b in n for b in BIG)]

def load(kind, name, **kw):
    """With the example values of any parameters the definition needs (a block size ...)"""
    ex = rp._config.examples(rp._config.read(rp.code.find(f"{kind}/{name}")))
    return rp.code.load(f"{kind}/{name}", **ex, **kw)

def test_kinds():
    assert set(rp.code.ls()) >= {"block/rs", "conv/k7_171_133", "interleave/forney"}

def test_unknown_definition():
    with pytest.raises(FileNotFoundError):
        rp.code.load("block/does.not.exist")

@pytest.mark.parametrize("name", configs("block"))
def test_block_round_trip(name, rng):
    code = load("block", name).encoder
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
    c = rp.code.load("block/hamming/7_4", decoder="linear").decoder
    msg = np.array([1, 0, 1, 1], dtype=bool)
    coded = c.encode(msg).copy()
    coded[2] ^= True
    assert np.array_equal(c.decode(coded), msg)

def test_rs_corrects_errors(rng):
    rs = rp.code.load("block/rs/204_188").encoder
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
    code = rp.code.load(f"conv/{name}")
    msg = bits(rng, 2400)
    out = np.asarray(code.decoder.decode(code.encoder.encode(msg))).astype(bool)
    n = min(len(out), len(msg)) - 40 # The tail of a non-terminated stream is not guaranteed
    assert n > 2000
    assert np.array_equal(out[:n], msg[:n])

def test_viterbi_soft_beats_hard(rng):
    code = rp.code.load("conv/k7_133_171/1_2")
    msg = bits(rng, 4000)
    coded = code.encoder.encode(msg)
    sigma = 0.75
    rx = (1.0 - 2.0*coded.astype(float)) + rng.normal(0, sigma, len(coded))
    hard = np.asarray(code.decoder.decode((rx < 0).astype(np.uint8))).astype(bool)
    soft = np.asarray(code.decoder.decode(rx, soft=True)).astype(bool)
    n = min(len(hard), len(soft), len(msg)) - 40
    assert (soft[:n] != msg[:n]).sum() < (hard[:n] != msg[:n]).sum()

def test_interleaver_round_trip(rng):
    il = rp.code.load("interleave/forney/12_17").encoder
    x = rng.integers(0, 256, 204*30).astype(np.uint8)
    y = il.decode(il.encode(x, symbols=True), symbols=True)
    delay = (il.I - 1)*il.M*il.I # Total latency of interleaver + deinterleaver
    assert np.array_equal(np.asarray(y)[delay:delay+1000], x[:1000])

def test_ldpc_corrects_noise(rng):
    code = rp.code.load("block/ldpc/gallager_96_48").encoder
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
    assert Path(path).is_file()
    rp.add_config_path(tmp_path / "cfg")
    try:
        code = rp.code.load("block/ldpc/alist_ham").encoder
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

@pytest.mark.parametrize("ref", ["block/ldpc/gallager_96_48", "block/ldpc/ieee80211n_648_1_2"])
def test_ldpc_flooding_schedule_decodes_what_layered_does(ref, rng):
    layered = rp.code.load(ref)
    flooding = rp.code.init("flood", {**rp._config.read(rp._config.find("fec", ref)), "schedule": "flooding", "iters": 100})
    k = layered.encoder.num
    msg = rng.integers(0, 2, 8*k).astype(bool)
    x = layered.encoder.encode(msg)
    llr = 2*(1 - 2.0*x)/0.6**2 + rng.normal(0, 2/0.6, len(x))
    a, b = layered.decoder.decode(llr), flooding.decoder.decode(llr)
    assert np.array_equal(a, msg) and np.array_equal(b, msg)
    with pytest.raises(ValueError):
        rp.code.init("x", {**rp._config.read(rp._config.find("fec", ref)), "schedule": "sideways"})

def test_rs_dual_basis_is_the_ccsds_one(rng):
    """CCSDS 131.0-B Annex F: the conventional-to-dual table starts 00 7b af d4 99 e2 36 4d (libfec's Taltab)"""
    dual = rp.code.load("block/rs/ccsds_255_223_dual")
    conv = rp.code.load("block/rs/ccsds_255_223")
    assert list(dual.encoder._to_dual[:8]) == [0x00, 0x7b, 0xaf, 0xd4, 0x99, 0xe2, 0x36, 0x4d]
    msg = rng.integers(0, 2, 2*223*8).astype(bool)
    cw = dual.encoder.encode(msg)
    as_conv = dual.encoder.to_bits(dual.encoder.to_symbols(cw)) # (round trip of the conversion)
    assert np.array_equal(as_conv, cw)
    assert not np.array_equal(cw, conv.encoder.encode(msg)) # the parity differs: not the same bits on air
    bad = cw.copy()
    for w in range(2): # 8 symbol errors a codeword (t = 16)
        bad[(w*255 + rng.choice(255, 8, replace=False))*8 + rng.integers(0, 8, 8)] ^= True
    assert np.array_equal(dual.decoder.decode(bad), msg)

@pytest.mark.parametrize("r, m", [(1, 5), (2, 4), (2, 5), (3, 6), (0, 3)])
def test_reed_muller_corrects_half_its_distance(r, m, rng):
    c = rp.code.init("rm", {"base": "block", "type": "rm", "r": r, "m": m})
    k, n = c.encoder.num, c.encoder.den
    assert n == 2**m
    msg = rng.integers(0, 2, 20*k).astype(bool)
    x = c.encoder.encode(msg)
    t = 2**(m - r - 1) - 1 if m > r else 0
    bad = x.reshape(20, n).copy()
    for w in range(20):
        bad[w, rng.choice(n, t, replace=False)] ^= True
    assert np.array_equal(c.decoder.decode(bad.reshape(-1)), msg)

def test_reed_muller_first_order_is_soft(rng):
    c = rp.code.load("block/rm/1_5")
    msg = rng.integers(0, 2, 50*6).astype(bool)
    x = c.encoder.encode(msg)
    llr = 2*(1 - 2.0*x) + rng.normal(0, 2, len(x)) # Es/N0 0 dB: hard decisions have ~16% errors
    assert np.mean(c.decoder.decode(llr) != msg) < 0.01

def _tpc_words(code, W, ebn0, seed):
    """W random messages through the code and BPSK over AWGN at Eb/N0 [dB]: (message, LLRs)"""
    rng = np.random.default_rng(seed)
    k, n = code.encoder.num, code.encoder.den
    msg = rng.integers(0, 2, W*k).astype(bool)
    x = code.encoder.encode(msg)
    sigma = np.sqrt(n/k/(2*10**(ebn0/10)))
    return msg, 2*((1 - 2.0*x) + rng.normal(0, sigma, len(x)))/sigma**2

@pytest.mark.parametrize("ebn0, soft_max", [(3.25, 2e-3), (3.75, 3e-4)])
def test_turbo_product_code_soft_beats_hard(ebn0, soft_max):
    """Chase-Pyndiah on LLRs decodes the (64,57)^2 code far below the hard iterative decoder. 32 words (104k bits) per point;
    over 20 seeds (chase 4, 4 iterations): soft at most 1.2e-4 at 3.25 dB and no errors at 3.75 dB; hard 1.8e-2 .. 3.6e-2.
    The AFF3CT reference (chase 5, 8 iterations) is 1.2e-3 at 2.75 dB and 1.2e-5 at 3.0 dB (tests/tpc_ber.py)"""
    tpc = rp.code.load("block/product/tpc_64_57_sq")
    hard = rp.code.init("hard", {**rp._config.read(rp._config.find("fec", "block/product/tpc_64_57_sq")), "chase": None})
    msg, llr = _tpc_words(tpc, 32, ebn0, 7)
    ber_soft = np.mean(tpc.decoder.decode(llr) != msg)
    ber_hard = np.mean(hard.decoder.decode(llr) != msg)
    assert ber_soft < soft_max and ber_hard > 1e-2
    assert ber_soft < ber_hard/20
    assert np.array_equal(tpc.decoder.decode(4.0*(1 - 2.0*(tpc.encoder.encode(msg)))), msg) and (tpc.decoder.errors == 0).all()

def test_turbo_product_code_iterations_do_not_hurt():
    """Near the threshold (3.0 dB) each doubling of the iterations lowers BER and FER: measured 1.1e-2 / 1.2e-3 / 2.3e-4
    for 2 / 4 / 8 iterations (20 seeds: each step at most 0.3 of the one before). Extrinsic information that fights the channel makes more iterations worse instead"""
    tpc = rp.code.load("block/product/tpc_64_57_sq")
    dec = tpc.decoder
    msg, llr = _tpc_words(tpc, 32, 3.0, 11)
    ber, fer = [], []
    for it in (2, 4, 8):
        dec.iters = it
        e = (dec.decode(llr) != msg).reshape(32, -1)
        ber.append(e.mean())
        fer.append(e.any(axis=1).mean())
    assert ber[1] < 0.5*ber[0] and ber[2] < 0.5*ber[1] + 1e-5
    assert fer[0] >= fer[1] >= fer[2]
