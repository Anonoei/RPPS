"""Blind tools for what no definition covers: feedthrough scramblers (rpps.analysis.scrambler), Reed-Solomon codes
(rpps.analysis.rs), the layers inside a decoded stream (fec.solve), and what must not be taken for a code"""
import numpy as np
import pytest

import rpps as rp
from rpps import _config
from rpps.analysis import fec as F, rs as B, scrambler as SC
from rpps.code.types.rs import rs
from rpps.logger import Logger

def scramble(d, taps):
    """y[n] = d[n] ^ y[n-t] for each tap (a feedthrough scrambler, register starting at zero)"""
    y = np.zeros(len(d), np.uint8)
    for n in range(len(d)):
        v = int(d[n])
        for t in taps:
            if n >= t:
                v ^= int(y[n - t])
        y[n] = v
    return y

@pytest.fixture(scope="module")
def hdlc():
    rng = np.random.default_rng(1)
    fr = rp.link.load("frame", "hdlc")
    frames = [bytes(rng.integers(0, 256, int(rng.integers(20, 200))).astype(np.uint8)) for _ in range(120)]
    return np.asarray(fr.encoder.encode(frames)).astype(np.uint8)[:100000]

def rs_bits(n, k, poly, fcr, prim, words=24, payload=None, seed=0, phase=37, offset=5):
    rng = np.random.default_rng(seed)
    code = rs(Logger().Child("t", 0), k, n, 8, poly, fcr, prim)
    msg = rng.integers(0, 256, (words, k)) if payload is None else np.packbits(payload[:words*k*8]).reshape(words, k)
    cw = np.asarray(code.encode(np.asarray(msg).reshape(-1), symbols=True)).astype(np.uint8)[phase:]
    return np.concatenate((rng.integers(0, 2, offset).astype(np.uint8), np.unpackbits(cw)))

def test_descramble_is_the_feedthrough_definition():
    y = np.random.default_rng(0).integers(0, 2, 2000).astype(bool)
    d = np.asarray((rp.Data(y, flush=True) / rp.code.scram.load("feedthrough/v35")).data).astype(np.uint8)
    assert np.array_equal(SC.descramble(y, (20, 3))[20:], d[20:])

def test_scrambler_found_from_the_structure_under_it(hdlc):
    h, n, g = SC.search(scramble(hdlc, (20, 3)))
    assert h[0].params["taps"] == [20, 3] and g.definition["poly"] == [20, 3]
    h, n, g = SC.search(scramble(np.ones(60000, np.uint8), (17, 12))) # An idle line of ones: the strongest structure
    assert g.definition["poly"] == [17, 12]

def test_three_taps(hdlc):
    assert not SC.search(scramble(hdlc, (23, 18, 5)))[0]
    assert SC.search(scramble(hdlc, (23, 18, 5)), max_taps=3)[2].definition["poly"] == [23, 18, 5]

def test_white_stays_white(hdlc):
    rng = np.random.default_rng(2)
    for y in (rng.integers(0, 2, 100000).astype(np.uint8), scramble(rng.integers(0, 2, 60000).astype(np.uint8), (20, 3))):
        h, n, g = SC.search(y)
        assert not h and g is None
    h, n, g = SC.search(hdlc) # Structure with no scrambler: "none" stands out, nothing to save
    assert h[0].params["taps"] == [] and g is None

def test_blind_rs_finds_field_generator_alignment_and_polarity():
    bits = rs_bits(219, 201, 0x187, 120, 1) ^ 1
    h, n, g = B.search(bits, lengths=(204, 219))
    p = h[0].params
    assert (p["n"], p["k"], p["poly"], p["fcr"], p["prim"], p["polarity"], p["bit_offset"]) == (219, 201, 0x187, 120, 1, 1, 5)
    assert g.definition == {**g.definition, "n": 219, "k": 201, "poly": 0x187, "fcr": 120, "prim": 1}

def test_blind_rs_generator_with_a_root_spacing():
    roots = [(11*(112 + i)) % 255 for i in range(32)]
    assert B.fit_generator(roots) == (112, 11)
    assert B.fit_generator([3, 7, 100]) is None

def test_blind_rs_nothing_in_random_or_constant_bits():
    rng = np.random.default_rng(3)
    assert not B.search(rng.integers(0, 2, 60000).astype(np.uint8), lengths=(204, 255))[0]
    assert not B.search(np.zeros(60000, np.uint8), lengths=(204, 255))[0] # Zero words are codewords of any RS code

def test_guesses_save_as_definitions_that_load(tmp_path, monkeypatch, hdlc):
    monkeypatch.setattr(_config, "_EXTRA", [tmp_path])
    y = scramble(hdlc, (20, 3))
    g = SC.search(y)[2]
    g.save("found", tmp_path/"scramble")
    sc = rp.code.scram.load("feedthrough/guess/found")
    d = np.asarray((rp.Data(y.astype(bool), flush=True) / sc).data).astype(np.uint8)
    assert np.array_equal(d[20:], hdlc[20:])
    bits = rs_bits(219, 201, 0x187, 120, 1, phase=0, offset=0)
    g = B.search(bits, lengths=(219,))[2]
    g.save("found", tmp_path/"fec")
    code = rp.code.load("block", "guess/found").encoder
    words = np.packbits(bits[:len(bits)//(8*219)*8*219]).reshape(-1, 219).astype(np.int64)
    assert not code.syndromes(words).any()

def test_solve_peels_scrambler_then_rs(hdlc):
    y = scramble(rs_bits(219, 201, 0x187, 120, 1, words=40, payload=hdlc, phase=0, offset=0), (20, 3))
    layers = F.solve(y, rs_lengths=(219,))
    assert [h.kind for h, g in layers] == ["scrambler", "rs-blind"]
    assert layers[0][1].definition["poly"] == [20, 3] and layers[1][1].definition["k"] == 201

def test_solve_finds_rs_of_random_payload_behind_a_known_scrambler():
    y = scramble(rs_bits(219, 201, 0x187, 120, 1, words=40, phase=0, offset=0), (20, 3))
    layers = F.solve(y, rs_lengths=(219,))
    assert layers[0][0].params.get("config") == "v35" and layers[1][1].definition["n"] == 219

def test_an_unmodulated_carrier_is_not_a_code():
    rng = np.random.default_rng(4)
    s = (np.ones(40000) + 0.3*(rng.standard_normal(40000) + 1j*rng.standard_normal(40000))).astype(np.complex64)
    assert not F.identify(s, "bpsk").found

def test_acquire_fec_says_it_is_still_acquiring():
    from rpps.analysis.modulation import Detection
    d = rp.Data(np.ones(1000, np.complex64), Fs=1e4, sps=1)
    d.meta.stream.det = Detection(mod="qpsk", baud=1e4, sps=1.0)
    d + rp.sync.AcquireFEC()
    f = d.meta.stream.est["fec"]
    assert not f.found and not f.complete and f.n == 1000

# --- Binary block codes, products, unknown puncture patterns ---------------------------------------------------------

from rpps.analysis import block as BK, puncture as PU
from rpps.code.types.bchgf import bchgf
from rpps.code.types.conv import conv
from rpps.math.gf2 import nullspace, rank, rref

def test_packed_gf2_matches_its_definition():
    rng = np.random.default_rng(0)
    for R, n in ((5, 9), (40, 64), (100, 130), (70, 70)):
        M = rng.integers(0, 2, (R, n)).astype(np.uint8)
        N = nullspace(M)
        A, piv = rref(M)
        assert not ((M.astype(int) @ N.T) % 2).any() and rank(M) == len(piv) == n - len(N)

def test_bch_found_with_its_field_and_t_shortened_too():
    log = Logger().Child("t", 0)
    rng = np.random.default_rng(1)
    for n, k in ((63, 51), (55, 43)):
        b = np.asarray(bchgf(log, n, k, 6, 0x43, 2).encode(rng.integers(0, 2, k*600).astype(bool))).astype(np.uint8)[3:]
        (h, g), = BK.solve_block(b)
        assert g.definition["type"] == "bch" and (g.definition["n"], g.definition["k"], g.definition["t"]) == (n, k, 2)
        assert g.definition["poly"] == "0x43" and h.params["offset"] == n - 3

def test_cyclic_code_found_by_its_generator():
    rng = np.random.default_rng(2)
    b = np.asarray(rp.code.load("block", "golay/23_12").encoder.encode(rng.integers(0, 2, 12*600).astype(bool))).astype(np.uint8)
    (h, g), = BK.solve_block(b[5:])
    assert g.definition["type"] == "cyclic" and int(g.definition["poly"], 16) == 0xC75

def test_an_ldpc_code_from_its_sparse_checks():
    rng = np.random.default_rng(6)
    H = np.asarray(rp.code.load("block", "ldpc/gallager_96_48").encoder.H, dtype=np.uint8)[:, rng.permutation(96)]
    cw = (rng.integers(0, 2, (1500, len(nullspace(H)))) @ nullspace(H) % 2).astype(np.uint8).reshape(-1)[13:]
    (h, g), = BK.solve_block(cw[:32768])
    assert h.params["kind"] == "ldpc" and (h.params["n"], h.params["k"]) == (96, 50)
    assert {len(c) for c in g.definition["checks"]} == {6} # Gallager's column-weight-3, row-weight-6 checks

def test_soft_values_find_sparse_checks_at_a_bit_error_rate_hard_bits_cannot():
    rng = np.random.default_rng(6)
    H = np.asarray(rp.code.load("block", "ldpc/gallager_96_48").encoder.H, dtype=np.uint8)[:, rng.permutation(96)]
    G = nullspace(H)
    cw = (rng.integers(0, 2, (3000, len(G))) @ G % 2).astype(np.uint8)
    y = (1 - 2.0*cw) + 0.4*rng.standard_normal(cw.shape) # BER about 0.6%
    S, _ = BK.sparse_checks((y < 0).astype(np.uint8), np.abs(y), k=50)
    assert rank(BK.lightest(S)) == 46

def test_product_code_rows_and_columns():
    rng = np.random.default_rng(3)
    r, c = rp.code.load("block", "hamming/15_11").encoder, rp.code.load("block", "hamming/7_4").encoder
    blocks = []
    for _ in range(500):
        rows = np.array([r.encode(x) for x in rng.integers(0, 2, (4, 11)).astype(bool)])
        blocks.append(np.array([c.encode(col) for col in rows.T]).T.reshape(-1))
    (h, g), = BK.solve_block(np.concatenate(blocks).astype(np.uint8)[29:])
    assert h.params["kind"] == "product" and (h.params["n1"], h.params["k1"], h.params["n2"], h.params["k2"]) == (15, 11, 7, 4)

def test_no_block_code_in_random_or_constant_bits():
    assert BK.solve_block(np.random.default_rng(4).integers(0, 2, 40000).astype(np.uint8)) == []
    assert BK.solve_block(np.ones(40000, np.uint8)) == []
    pattern = np.random.default_rng(5).integers(0, 2, 128).astype(np.uint8) # Idle fill repeating every 128 bits
    assert BK.solve_block(np.tile(pattern, 400)[37:]) == []

def test_unknown_puncture_pattern_on_a_known_mother_code():
    rng = np.random.default_rng(2)
    g = [[1, 0, 1, 1, 0, 1, 1], [1, 1, 1, 1, 0, 0, 1]]
    pat = [[1, 0, 1, 0, 1, 1], [1, 1, 0, 1, 0, 0]] # Rate 6/7, in no definition
    bits = conv(Logger().Child("t", 0), 1, 2, g, 7, pat).encode(rng.integers(0, 2, 20000).astype(bool))[3:]
    q = rp.symbol.load("qpsk")
    s = q.modulate(bits[:len(bits)//2*2])*1j
    s = s + 0.15*(rng.standard_normal(len(s)) + 1j*rng.standard_normal(len(s)))
    v, guess = PU.best(*PU.search(F.bit_streams(s[:8000], q, differential=False)))
    found = np.array(guess.definition["encode"]["puncture"], dtype=bool)
    assert any(np.array_equal(found, np.roll(np.array(pat, dtype=bool), sh, axis=1)) for sh in range(6)) # (up to a turn)
    assert v.params["mismatch"] < 0.01
    q2 = q.points[rng.integers(0, 4, 8000)] + 0.1*(rng.standard_normal(8000) + 1j*rng.standard_normal(8000))
    assert PU.best(*PU.search(F.bit_streams(q2, q, differential=False))) == (None, None)

def test_blind_identify_peels_a_punctured_code_and_the_bch_inside():
    rng = np.random.default_rng(5)
    log = Logger().Child("t", 0)
    inner = bchgf(log, 63, 51, 6, 0x43, 2).encode(rng.integers(0, 2, 51*1200).astype(bool))
    outer = conv(log, 1, 2, [[1, 0, 1, 1, 0, 1, 1], [1, 1, 1, 1, 0, 0, 1]], 7, [[1, 1, 1, 1], [1, 0, 0, 0]]).encode(inner.astype(bool))[5:]
    q = rp.symbol.load("qpsk")
    s = (q.modulate(outer[:len(outer)//2*2])*1j + 0.12*(rng.standard_normal(len(outer)//2) + 1j*rng.standard_normal(len(outer)//2))).astype(np.complex64)
    f = F.identify(s, "qpsk", blind=True)
    assert not f.found and [h.kind for h in f.inner] == ["conv-punctured", "block-blind"]
    assert f.guesses[1].definition["type"] == "bch"
