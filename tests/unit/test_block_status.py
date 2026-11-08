"""Per-codeword decode status, syndrome-table / cyclic codes (Golay), Reed-Solomon erasures and puncturing"""
import itertools

import numpy as np
import pytest

import rpps as rp
from conftest import bits

# Received on air (P25 phase 1, 853.575 MHz): the 24 hexbits of an LDU1 (link control: group voice, talkgroup 2813,
# source 0, service options 0x04) and of an LDU2 (encryption sync: algorithm 0x80, key 0, message indicator 0)
LDU1 = [0, 0, 0, 4, 0, 0, 43, 61, 0, 0, 0, 0, 45, 4, 5, 48, 29, 30, 38, 22, 6, 21, 11, 11]
LDU2 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 32, 0, 0, 0, 43, 11, 34, 36, 38, 61, 49, 53]

# --- Reed-Solomon ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("ref,word", [("rs/p25_24_12", LDU1), ("rs/p25_24_16", LDU2)])
def test_rs_p25_on_air_words(ref, word, rng):
    code = rp.coding.load("block", ref)._dec
    w = np.array(word)
    assert not code.syndromes(w[None]).any() # A real codeword: the definition's roots are right
    bad = w.copy()
    pos = rng.permutation(24)[:code.t]
    bad[pos] ^= rng.integers(1, 64, len(pos))
    out = code.decode(bad, symbols=True)
    assert np.array_equal(out, w[:code.num]) and code.errors[0] == code.t

def _rs_trial(code, rng, n_err, n_era):
    info = rng.integers(0, 2**code.m, code.num)
    cw = code.encode(info, symbols=True)
    pos = rng.permutation(code.den)
    era, err = pos[:n_era], pos[n_era:n_era + n_err]
    rx = cw.copy()
    rx[err] ^= rng.integers(1, 2**code.m, n_err)
    rx[era[:n_era//2]] ^= rng.integers(1, 2**code.m, n_era//2) # Half the erased symbols are wrong too
    out = code.decode(rx, symbols=True, erasures=[era])
    return np.array_equal(out, info), int(code.errors[0])

def test_rs_errors_and_erasures(rng):
    code = rp.coding.load("block", "p25p2_63_35")._dec
    for _ in range(60):
        n_era = int(rng.integers(0, code.nroots + 1))
        n_err = int(rng.integers(0, (code.nroots - n_era)//2 + 1))
        ok, fixed = _rs_trial(code, rng, n_err, n_era)
        assert ok and fixed == n_err + n_era//2

def test_rs_beyond_capacity_fails_cleanly(rng):
    code = rp.coding.load("block", "p25p2_63_35")._dec
    fails = [_rs_trial(code, rng, 10, 12)[1] for _ in range(20)] # 2*10 + 12 > 28
    assert fails.count(-1) >= 18

def test_rs_erasure_mask_form(rng):
    code = rp.coding.RS(24, 12, m=6, fcr=1)._dec
    info = rng.integers(0, 64, (3, 12))
    cw = code.encode(info.reshape(-1), symbols=True).reshape(3, 24)
    mask = np.zeros((3, 24), bool)
    mask[:, :12] = True # 12 erasures: the whole capacity
    rx = cw.copy()
    rx[:, :12] = 0
    assert np.array_equal(code.decode(rx.reshape(-1), symbols=True, erasures=mask), info.reshape(-1))
    assert (code.errors >= 0).all()

@pytest.mark.parametrize("ref,sent", [("p25p2_soemi", 45), ("p25p2_ioemi", 52)])
def test_rs_punctured(ref, sent, rng):
    """Shortened and punctured as P25 phase 2 sends its MAC: the missing parity is decoded as erasures"""
    code = rp.coding.load("block", ref)
    assert code._dec.den == sent
    msg = bits(rng, 6*code._dec.num)
    cw = (rp.Data(msg, flush=True) * code).data
    assert len(cw) == 6*sent
    t = (code._dec.nroots - len(code._dec.puncture))//2
    rx = cw.copy().reshape(-1, 6)
    rx[rng.permutation(sent)[:t], 0] ^= True
    out = rp.Data(rx.reshape(-1), flush=True) / code
    d = code._dec
    full = rp.coding.RS(d.n, d.num, m=6, fcr=1)._enc.encode(msg, symbols=False) # The same word, nothing punctured
    restored = np.count_nonzero(d.to_symbols(full)[d.puncture]) # Punctured symbols the decoder had to fill in
    assert np.array_equal(out.data, msg) and code.errors[0] == t + restored

def test_rs_soft_retry_erases_unreliable(rng):
    code = rp.coding.load("block", "p25p2_63_35")._dec
    info = rng.integers(0, 64, 35)
    b = code.to_bits(code.encode(info, symbols=True))
    llr = np.where(b, -4.0, 4.0)
    for p in rng.permutation(63)[:20]: # 20 wrong symbols (t = 14), each received weakly
        llr[6*p:6*p + 6] *= -0.1
    out = code.decode(llr)
    assert np.array_equal(code.to_symbols(out), info) and code.errors[0] == 20
    code.decode(llr < 0, soft=False)
    assert code.errors[0] == -1 # The same word from hard bits is beyond errors-only decoding

def test_rs_dvb_unchanged(rng):
    code = rp.coding.load("block", "rs/204_188")
    msg = bits(rng, 8*188*2)
    cw = (rp.Data(msg, flush=True) * code).data.copy()
    cw[:64] ^= True # 8 byte errors in the first word
    assert np.array_equal((rp.Data(cw, flush=True) / code).data, msg)
    assert list(code.errors) == [8, 0]

# --- cyclic codes, syndrome decoder -------------------------------------------------------------------------

def test_golay_corrects_every_three_error_pattern(rng):
    code = rp.coding.Golay(23)
    msg = bits(rng, 12)
    cw = code._enc.encode(msg)
    pats = list(itertools.combinations(range(23), 3))
    for idx in rng.permutation(len(pats))[:300]:
        rx = cw.copy()
        rx[list(pats[idx])] ^= True
        assert np.array_equal(code._dec.decode(rx), msg) and code.errors[0] == 3

def test_golay23_is_perfect():
    dec = rp.coding.Golay(23)._dec
    assert (dec.weight >= 0).all() # Every syndrome is within 3 errors of a codeword

def test_extended_golay_detects_four(rng):
    code = rp.coding.load("block", "golay/24_12")
    assert code._dec.t == 3
    msg = bits(rng, 12)
    cw = code._enc.encode(msg)
    assert cw.sum() % 2 == 0 # Overall parity
    for _ in range(50):
        rx = cw.copy()
        rx[rng.permutation(24)[:4]] ^= True
        code._dec.decode(rx)
        assert code.errors[0] == -1

def test_cyclic_shortened(rng):
    code = rp.coding.Cyclic(23, 12, 0xC75, extend=True, shorten=6) # (18,6,8)
    assert (code._enc.num, code._enc.den, code._dec.t) == (6, 18, 3)
    msg = bits(rng, 6*20)
    cw = (rp.Data(msg, flush=True) * code).data.copy().reshape(-1, 18)
    for w in cw:
        w[rng.permutation(18)[:3]] ^= True
    assert np.array_equal((rp.Data(cw.reshape(-1), flush=True) / code).data, msg)
    assert (code.errors == 3).all()

def test_cyclic_rejects_non_generator():
    with pytest.raises(ValueError):
        rp.coding.Cyclic(23, 12, 0xC77)
    with pytest.raises(ValueError):
        rp.coding.Cyclic(23, 13, 0xC75)

def test_cyclic_matches_bch():
    """BCH (15,7,5) by its generator: x^8+x^7+x^6+x^4+1, two errors"""
    code = rp.coding.Cyclic(15, 7, 0b111010001)
    assert code._dec.t == 2

# --- status of the other block decoders ---------------------------------------------------------------------

def test_linear_own_decoder_flags_instead_of_raising(rng):
    code = rp.coding.load("block", "hamming/p25_10_6", decoder="linear")
    msg = bits(rng, 6*40)
    cw = code._enc.encode(msg).reshape(-1, 10).copy()
    cw[::2, [0, 3]] ^= True # Syndrome 1110 ^ 0111 = 1001: no single flip makes it
    code._dec.decode(cw.reshape(-1))
    assert list(code.errors) == [-1, 0]*20

def test_trellis_decoder_status(rng):
    code = rp.coding.load("block", "hamming/7_4")
    msg = bits(rng, 4*30)
    cw = code._enc.encode(msg).reshape(-1, 7).copy()
    cw[::2, 3] ^= True
    out = code._dec.decode(cw.reshape(-1))
    assert np.array_equal(out, msg)
    assert list(code.errors) == [1, 0]*15

def test_repeat_status():
    rep = rp.coding.load("block", "repeat/3")._dec
    rep.decode(np.array([1, 1, 0, 0, 0, 0, 1, 1, 1], dtype=np.uint8))
    assert list(rep.errors) == [1, 0, 0]

def test_ldpc_status(rng):
    code = rp.coding.load("block", "ldpc/gallager_96_48")
    msg = bits(rng, code._enc.num*4)
    cw = code._enc.encode(msg)
    llr = (1.0 - 2.0*cw)*4.0
    llr[5] *= -1
    code._dec.decode(llr)
    assert list(code.errors) == [1, 0, 0, 0]

def test_crc_status(rng):
    crc = rp.encap.load("crc", "crc16_xmodem")
    sent = crc._enc.encode(bits(rng, 40))
    crc._enc.decode(sent)
    assert list(crc.errors) == [0]
    sent[3] ^= True
    crc._enc.decode(sent)
    assert list(crc.errors) == [-1]

def test_conv_has_no_word_status():
    assert rp.coding.load("conv", "k7_133_171/1_2").errors is None
