"""Soft decisions: LLRs from the demodulator, soft-input decoders, and the trellis (BCJR) decoders"""
import itertools

import numpy as np
import pytest

import rpps as rp
from rpps.coding.types import Trellis, bcjr
from conftest import bits, awgn

def modem(name):
    m = rp.mod.load(name)
    m.set_mapping(m.get_maps()[0])
    return m

# --- LLRs from the demodulator ---------------------------------------------------------------------

def test_bpsk_llr_is_closed_form(rng):
    m = modem("bpsk")
    msg = bits(rng, 2000)
    y, var = awgn(m.constellation.modulate(msg), 3.0, rng)
    llr = m.constellation.llr(y, var)
    sign = 1.0 if m.points.arr[m.constellation.mapping.arr.tolist().index(0)].real > 0 else -1.0
    assert np.allclose(llr, sign*4*y.real/var, rtol=1e-4, atol=1e-4)

@pytest.mark.parametrize("name", ["bpsk", "qpsk", "8psk", "dvbs2_8psk"])
def test_llr_sign_is_the_hard_decision(name, rng):
    m = modem(name)
    msg = bits(rng, 3*4000)
    y, var = awgn(m.constellation.modulate(msg), 24.0, rng)
    llr = m.constellation.llr(y, var)
    assert len(llr) == len(msg)
    assert ((llr < 0) == msg).mean() > 0.999
    maxlog = m.constellation.llr(y, var, maxlog=True)
    assert ((maxlog < 0) == (llr < 0)).mean() > 0.999

@pytest.mark.parametrize("name", ["bpsk", "qpsk", "8psk"])
def test_noise_estimate(name, rng):
    m = modem(name)
    y, var = awgn(m.constellation.modulate(bits(rng, 3*20000)), 8.0, rng)
    assert m.constellation.estimate_noise(y) == pytest.approx(var, rel=0.1)

def test_llr_is_calibrated(rng):
    """Among bits given |LLR| about L, the error rate is 1/(1+e^L)"""
    m = modem("qpsk")
    msg = bits(rng, 2*60000)
    y, var = awgn(m.constellation.modulate(msg), 2.0, rng)
    llr = m.constellation.llr(y, var)
    for lo, hi in ((0.5, 1.0), (1.5, 2.5)):
        sel = (np.abs(llr) >= lo) & (np.abs(llr) < hi)
        want = np.mean(1/(1 + np.exp(np.abs(llr[sel]))))
        got = np.mean((llr[sel] < 0) != msg[sel])
        assert got == pytest.approx(want, abs=0.02)

def test_demod_gives_soft_mod(rng):
    m = modem("qpsk")
    syms = rp.Data(bytes(rng.integers(0, 256, 8).astype(np.uint8))).as_bits() * m
    out = syms / m
    assert out.is_soft and out.get_soft().dtype == np.float32
    assert out.get_soft(hard_llr=1.0) is out.get_soft()
    hard = out.as_hard()
    assert not hard.is_soft and hard.get_soft() is None and hard.get_soft(hard_llr=2.0) is not None

# --- Trellis decoders ------------------------------------------------------------------------------

def brute_app(codewords, llr, maxlog):
    """A-posteriori LLRs of each coded bit by enumerating every codeword"""
    codewords = codewords.astype(np.int64)
    score = ((1 - 2*codewords)*llr/2).sum(axis=1)
    reduce = (lambda v: v.max()) if maxlog else np.logaddexp.reduce
    return np.array([reduce(score[codewords[:, j] == 0]) - reduce(score[codewords[:, j] == 1]) for j in range(codewords.shape[1])])

@pytest.mark.parametrize("name", ["hamming/7_4", "hamming/15_11", "hamming/3_1"])
@pytest.mark.parametrize("maxlog", [False, True])
def test_trellis_matches_enumeration(name, maxlog, rng):
    code = rp.coding.load("block", name)
    enc, dec = code._enc, code._dec
    assert type(dec).__name__ == "trellis"
    dec.maxlog = maxlog
    msgs = np.array(list(itertools.product([0, 1], repeat=enc.num)), dtype=bool)
    cw = enc.encode(msgs.reshape(-1)).reshape(len(msgs), enc.den).astype(np.uint8)
    llr = rng.normal(0, 2, (4, enc.den))
    app = dec.app(llr.reshape(-1))
    for w in range(4):
        assert np.allclose(app[w], brute_app(cw, llr[w], maxlog), atol=1e-9)

def test_hamming_corrects_any_single_error():
    code = rp.coding.load("block", "hamming/7_4")
    msg = np.array([1, 0, 1, 1, 0, 0, 1, 0], dtype=bool)
    coded = code._enc.encode(msg)
    for i in range(7):
        bad = coded.copy()
        bad[i] ^= True
        assert np.array_equal(code._dec.decode(bad), msg)

def test_soft_trellis_beats_hard_on_hamming(rng):
    code = rp.coding.load("block", "hamming/7_4")
    msg = bits(rng, 4*4000)
    coded = code._enc.encode(msg)
    rx = (1.0 - 2.0*coded) + rng.normal(0, 0.8, len(coded))
    hard = code._dec.decode((rx < 0))
    soft = code._dec.decode(2*rx/0.8**2)
    assert (soft != msg).sum() < (hard != msg).sum()
    assert (soft != msg).mean() < 0.06 # Uncoded BPSK at this noise level makes about 10.6% errors

def test_linear_decoder_override():
    code = rp.coding.load("block", "hamming/7_4", decoder="linear")
    assert type(code._dec).__name__ == "linear" and code.decision.name == "HARD"

def test_big_linear_code_falls_back():
    """A syndrome trellis of 2^(n-k) states is refused for long codes: the plain decoder takes over"""
    with pytest.raises(ValueError):
        Trellis.syndrome(np.zeros((17, 40), dtype=np.uint8))

TERMINATED = {
    "base": "conv", "type": "split",
    "encode": {"type": "conv", "num": 1, "den": 2, "generator": [[1, 0, 1], [1, 1, 1]], "constraint": 3, "terminated": True},
    "decode": {"type": "viterbi", "num": 1, "den": 2, "generator": [[1, 0, 1], [1, 1, 1]], "constraint": 3, "terminated": True},
}

@pytest.mark.parametrize("maxlog", [False, True])
def test_conv_bcjr_matches_enumeration(maxlog, rng):
    code = rp.coding.init("t", TERMINATED, decoder="bcjr")
    dec, K = code._dec, 6
    dec.maxlog = maxlog
    msgs = np.array(list(itertools.product([0, 1], repeat=K)), dtype=bool)
    cws = np.array([code._enc.encode(m) for m in msgs]).astype(np.uint8)  # (64, 2*(K+2))
    llr = rng.normal(0, 1.5, cws.shape[1])
    score = ((1 - 2*cws.astype(np.int64))*llr/2).sum(axis=1)
    reduce = (lambda v: v.max()) if maxlog else np.logaddexp.reduce
    want = np.array([reduce(score[msgs[:, j] == 0]) - reduce(score[msgs[:, j] == 1]) for j in range(K)])
    assert np.allclose(dec.app(llr), want, atol=1e-9)

def test_conv_bcjr_clean_and_hard(rng):
    for name in ("k3_5_7/1_2", "k7_133_171/1_2", "k7_171_133/3_4"):
        code = rp.coding.load("conv", name, decoder="bcjr")
        msg = bits(rng, 600)
        out = code._dec.decode(code._enc.encode(msg))
        n = min(len(out), len(msg)) - 20
        assert np.array_equal(out[:n], msg[:n])

def test_bcjr_close_to_viterbi_and_better_calibrated(rng):
    msg = bits(rng, 4000)
    code_v = rp.coding.load("conv", "k7_133_171/1_2")
    code_b = rp.coding.load("conv", "k7_133_171/1_2", decoder="bcjr")
    coded = code_v._enc.encode(msg)
    sigma = 0.9
    rx = (1.0 - 2.0*coded) + rng.normal(0, sigma, len(coded))
    llr = 2*rx/sigma**2
    v = code_v._dec.decode(llr)
    b = code_b._dec.decode(llr)
    n = min(len(v), len(b), len(msg)) - 20
    ev, eb = (v[:n] != msg[:n]).sum(), (b[:n] != msg[:n]).sum()
    assert eb <= ev*1.3 + 5 # BCJR minimises bit errors, Viterbi sequence errors: close
    app = code_b._dec.app(llr)[:n]
    wrong = (app < 0) != msg[:n]
    conf = np.abs(app) > 8
    assert wrong[conf].mean() < 1e-2 # Confident bits are right

def test_bcjr_handles_puncturing(rng):
    code = rp.coding.load("conv", "k7_171_133/3_4", decoder="bcjr")
    msg = bits(rng, 1200)
    coded = code._enc.encode(msg)
    rx = (1.0 - 2.0*coded) + rng.normal(0, 0.5, len(coded))
    out = code._dec.decode(2*rx/0.25)
    n = min(len(out), len(msg)) - 30
    assert (out[:n] != msg[:n]).mean() < 0.01

# --- soft decisions through the pipeline --------------------------------------------------------------

def test_decisions_default_by_decoder():
    assert rp.coding.load("conv", "k7_171_133/3_4").decision.name == "SOFT"
    assert rp.coding.load("block", "ldpc/gallager_96_48").decision.name == "SOFT"
    assert rp.coding.load("block", "repeat/3").decision.name == "SOFT"
    assert rp.coding.load("block", "hamming/7_4").decision.name == "SOFT"
    assert rp.coding.load("block", "rs/204_188").decision.name == "HARD"

def test_repeat_adds_llrs():
    rep = rp.coding.load("block", "repeat/3")._dec
    llr = np.array([+3.0, -1.0, -1.0,  +1.0, +1.0, -1.5], dtype=np.float32)
    assert rep.decode(llr).tolist() == [False, False]
    assert rep.decode(np.array([1, 1, 0, 0, 1, 0], dtype=bool)).tolist() == [True, False]

def run_link(ecc, mod, esn0, rng, nbits, soft=True):
    """bits -> ecc -> mod -> AWGN -> demod -> ecc, as BER of the message"""
    ecc.decision = rp.coding.coding.Decision.SOFT if soft else rp.coding.coding.Decision.HARD
    msg = bits(rng, nbits)
    syms = rp.Data(msg.copy()) * ecc * mod
    y, var = awgn(syms.data, esn0, rng)
    out = (rp.base.Analog(y, pp=rp.pproc.MOD, dt=rp.dtype.SYMBOLS) / mod) / ecc
    got = out.as_bits().data
    n = min(len(got), len(msg)) - 40
    return (got[:n] != msg[:n]).mean()

def test_soft_beats_hard_end_to_end_conv(rng):
    ecc, mod = rp.coding.load("conv", "k7_133_171/1_2"), modem("qpsk")
    hard = run_link(ecc, mod, 2.5, np.random.default_rng(5), 6000, soft=False)
    soft = run_link(ecc, mod, 2.5, np.random.default_rng(5), 6000, soft=True)
    assert soft < hard
    assert soft < 0.01

def test_soft_beats_hard_end_to_end_ldpc():
    ecc, mod = rp.coding.load("block", "ldpc/gallager_96_48"), modem("qpsk")
    n = ecc.e_num*60
    hard = run_link(ecc, mod, 2.0, np.random.default_rng(9), n, soft=False)
    soft = run_link(ecc, mod, 2.0, np.random.default_rng(9), n, soft=True)
    assert soft < hard

def test_hard_only_decoder_still_works_behind_soft_demod(rng):
    """RS can't use LLRs: the demodulator's soft output is decided before it"""
    ecc, mod = rp.coding.load("block", "rs/204_188"), modem("qpsk")
    msg = rp.Data(bytes(rng.integers(0, 256, 188*2).astype(np.uint8)))
    want = msg.as_bytes().hex
    out = (((msg*ecc*mod) / mod) / ecc).as_bytes().hex
    assert out == want
