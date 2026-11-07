import numpy as np
import pytest

import rpps as rp
from rpps.math import GF
from rpps.math.gf2 import nullspace, systematic
from rpps.math.prob import logsumexp, deviance_z, z_critical
from rpps.math.bits import reflect

@pytest.mark.parametrize("m,poly", [(4, 0x13), (8, 0x11D), (16, 0x1002D)])
def test_gf_field_axioms(m, poly, rng):
    gf = GF(m, poly)
    a, b, c = rng.integers(1, gf.size, (3, 200))
    assert np.array_equal(gf.mul(a, b), gf.mul(b, a))
    assert np.array_equal(gf.mul(gf.mul(a, b), c), gf.mul(a, gf.mul(b, c)))
    assert np.array_equal(gf.mul(a, b ^ c), gf.mul(a, b) ^ gf.mul(a, c)) # Addition is xor
    assert (gf.mul(a, gf.inv(a)) == 1).all()
    assert (gf.mul(a, 0) == 0).all()
    assert len(set(gf.exp[:gf.size-1].tolist())) == gf.size - 1 # alpha is primitive: poly is a primitive polynomial

def test_gf_pow_and_polymul():
    gf = GF(8, 0x11D)
    assert gf.pow(0, 3) == 0 and gf.pow(7, 0) == 1
    assert gf.pow(2, 8) == gf.mul(gf.pow(2, 4), gf.pow(2, 4))
    assert gf.pow_alpha(255) == 1
    # (x + 1)(x + 1) = x^2 + 1 in characteristic 2
    assert gf.polymul([1, 1], [1, 1]).tolist() == [1, 0, 1]
    a, b = np.array([3, 5, 7]), np.array([2, 9])
    x = 11
    ev = lambda p: np.bitwise_xor.reduce([gf.mul(c, gf.pow(x, len(p)-1-i)) for i, c in enumerate(p)])
    assert ev(gf.polymul(a, b)) == gf.mul(ev(a), ev(b))

def test_nullspace(rng):
    M = rng.integers(0, 2, (5, 12)).astype(np.uint8)
    N = nullspace(M)
    assert ((M @ N.T) % 2 == 0).all()
    assert len(N) >= 12 - 5
    assert np.array_equal(nullspace(np.zeros((3, 4), dtype=np.uint8)), np.eye(4, dtype=np.uint8))

def test_systematic_gives_generator_of_h(rng):
    H = rng.integers(0, 2, (6, 14)).astype(bool)
    G, info = systematic(H)
    assert ((H.astype(int) @ G.T.astype(int)) % 2 == 0).all()
    assert len(info) == G.shape[0]
    assert np.array_equal(G[:, info], np.eye(len(info), dtype=bool)) # Message bits appear in the codeword

def test_logsumexp(rng):
    x = rng.normal(size=(5, 7))
    assert np.allclose(logsumexp(x), np.log(np.exp(x).sum(axis=-1)))
    assert np.allclose(logsumexp(x, axis=0), np.log(np.exp(x).sum(axis=0)))
    assert np.isfinite(logsumexp(np.array([[1000.0, 1000.0]]))).all() # No overflow
    assert logsumexp(np.array([[-np.inf, -np.inf]]))[0] == -np.inf

def test_deviance_and_critical_z():
    assert deviance_z(50, 100) == pytest.approx(0)
    assert deviance_z(80, 100) > 5 and deviance_z(20, 100) < -5
    assert z_critical(1000) > z_critical(10) > 0

def test_reflect():
    assert reflect(0b0001, 4) == 0b1000
    assert reflect(0xA5, 8) == 0xA5
    assert reflect(0x01, 16) == 0x8000

def test_bch_corrects_up_to_t_errors(rng):
    code = rp.coding.load("block", "bch/dvbs2_1_2")._enc
    msg = rng.integers(0, 2, code.num).astype(bool)
    coded = code.encode(msg)
    bad = coded.copy()
    pos = rng.choice(code.den, code.t, replace=False)
    bad[pos] ^= True
    assert np.array_equal(code.decode(bad), msg)
    assert code.errors[0] == code.t
    bad = coded.copy()
    bad[rng.choice(code.den, code.t + 3, replace=False)] ^= True
    code.decode(bad)
    assert code.errors[0] == -1 # Beyond t: reported as uncorrectable

# --- Cyclic code decoding (shared by RS and BCH) ------------------------------------------------

def inject(words, errors, rng, symbols=True, m=8):
    """Corrupt `errors` distinct positions of every word"""
    bad = words.copy()
    for w in range(len(bad)):
        pos = rng.choice(bad.shape[1], errors, replace=False)
        bad[w, pos] ^= rng.integers(1, 2**m, errors).astype(bad.dtype) if symbols else 1
    return bad

RS_VARIANTS = {
    "dvb": dict(n=204, k=188, m=8, poly=0x11D, fcr=0, prim=1),
    "ccsds": dict(n=255, k=223, m=8, poly=0x187, fcr=112, prim=11),
    "gf16": dict(n=15, k=9, m=4, poly=0x13, fcr=1, prim=1),
    "short": dict(n=60, k=44, m=8, poly=0x11D, fcr=1, prim=1),
}

@pytest.mark.parametrize("name", RS_VARIANTS)
def test_rs_variants_correct_t_errors(name, rng):
    p = RS_VARIANTS[name]
    rs = rp.coding.init("t", {"base": "block", "type": "rs", **p})._enc
    msg = rng.integers(0, 2**p["m"], 20*p["k"]).astype(np.int64)
    cw = rs.encode(msg, symbols=True).reshape(20, p["n"])
    bad = inject(cw, rs.t, rng, m=p["m"])
    out = rs.decode(bad.reshape(-1), symbols=True)
    assert np.array_equal(np.asarray(out), msg)
    assert (rs.errors == rs.t).all()
    over = inject(cw, rs.t + 1, rng, m=p["m"]) # One too many: never a wrong "correction" reported as success
    rs.decode(over.reshape(-1), symbols=True)
    assert (rs.errors != rs.t + 1).all() # More than t is either flagged, or lands on another codeword with t or fewer
    assert (rs.errors == -1).mean() > 0.5

def test_rs_clean_words_untouched(rng):
    rs = rp.coding.load("block", "rs/204_188")._enc
    msg = rng.integers(0, 256, 188*3).astype(np.int64)
    assert np.array_equal(rs.decode(rs.encode(msg, symbols=True), symbols=True), msg)
    assert (rs.errors == 0).all()

@pytest.mark.parametrize("n,k,m,poly,t", [(31, 21, 5, 0x25, 2), (63, 45, 6, 0x43, 3), (15, 7, 4, 0x13, 2)])
def test_small_bch_codes(n, k, m, poly, t, rng):
    code = rp.coding.init("t", {"base": "block", "type": "bch", "n": n, "k": k, "m": m, "poly": hex(poly), "t": t})._enc
    msg = rng.integers(0, 2, 30*k).astype(bool)
    cw = code.encode(msg).reshape(30, n).astype(np.uint8)
    bad = inject(cw, t, rng, symbols=False)
    assert np.array_equal(code.decode(bad.reshape(-1)), msg)
    assert (code.errors == t).all()

def test_rs_decoding_is_fast(rng):
    import time
    rs = rp.coding.load("block", "rs/204_188")._enc
    cw = rs.encode(rng.integers(0, 256, 188*200).astype(np.int64), symbols=True).reshape(200, 204)
    bad = inject(cw, 8, rng)
    t0 = time.time()
    rs.decode(bad.reshape(-1), symbols=True)
    assert time.time() - t0 < 5 # The vectorised Chien search: about 0.3 s here, the scalar loop took 7 s
