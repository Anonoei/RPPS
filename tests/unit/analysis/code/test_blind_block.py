"""Blind block codes: RS, BCH, cyclic, LDPC and product codes"""
import numpy as np
import pytest

import rpps as rp
from rpps.analysis import identify as F
from rpps.analysis.code.block import rs as B
from rpps.logger import Logger
from rpps.analysis.code.block import linear as BK
from rpps.code.types.bchgf import bchgf
from rpps.code.types.conv import conv
from rpps.math.gf2 import nullspace, rank, rref
from support.blind import rs_bits, _qpsk

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
    b = np.asarray(rp.code.load("block/golay/23_12").encoder.encode(rng.integers(0, 2, 12*600).astype(bool))).astype(np.uint8)
    (h, g), = BK.solve_block(b[5:])
    assert g.definition["type"] == "cyclic" and int(g.definition["poly"], 16) == 0xC75

def test_an_ldpc_code_from_its_sparse_checks():
    rng = np.random.default_rng(6)
    H = np.asarray(rp.code.load("block/ldpc/gallager_96_48").encoder.H, dtype=np.uint8)[:, rng.permutation(96)]
    cw = (rng.integers(0, 2, (1500, len(nullspace(H)))) @ nullspace(H) % 2).astype(np.uint8).reshape(-1)[13:]
    (h, g), = BK.solve_block(cw[:32768])
    assert h.params["kind"] == "ldpc" and (h.params["n"], h.params["k"]) == (96, 50)
    assert {len(c) for c in g.definition["checks"]} == {6} # Gallager's column-weight-3, row-weight-6 checks

def test_soft_values_find_sparse_checks_at_a_bit_error_rate_hard_bits_cannot():
    rng = np.random.default_rng(6)
    H = np.asarray(rp.code.load("block/ldpc/gallager_96_48").encoder.H, dtype=np.uint8)[:, rng.permutation(96)]
    G = nullspace(H)
    cw = (rng.integers(0, 2, (3000, len(G))) @ G % 2).astype(np.uint8)
    y = (1 - 2.0*cw) + 0.4*rng.standard_normal(cw.shape) # BER about 0.6%
    S, _ = BK.sparse_checks((y < 0).astype(np.uint8), np.abs(y), k=50)
    assert rank(BK.lightest(S)) == 46

def test_product_code_rows_and_columns():
    rng = np.random.default_rng(3)
    r, c = rp.code.load("block/hamming/15_11").encoder, rp.code.load("block/hamming/7_4").encoder
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

@pytest.mark.kernels
def test_an_odd_length_block_code_on_qpsk_is_found_at_its_length():
    # BCH(63,51) on QPSK: a map that swaps the bits of a symbol cuts across words of odd length and still shows
    # structure at 126; of every map's find, the shortest code is the one
    rng = np.random.default_rng(7)
    b = bchgf(Logger().Child("t", 0), 63, 51, 6, 0x43, 2).encode(rng.integers(0, 2, 51*1100).astype(bool))[5:]
    f = F.identify(_qpsk(b, rng)[:2**15], "qpsk", blind=True)
    h = f.inner[0]
    assert (h.params["n"], h.params["k"]) == (63, 51) and f.guesses[0].definition["type"] == "bch"

@pytest.mark.kernels
def test_soft_scoring_finds_a_configured_code_hard_bits_cannot():
    # K=7 rate 1/2 on QPSK at Es/N0 3 dB: the hard decisions are ~8% wrong, near what a wrong code gets on random bits,
    # so the hard re-encoding test can't take it; weighted by |LLR| the right code's disagreements fall on weak bits
    from rpps.analysis.code.conv import mother as M
    rng = np.random.default_rng(13)
    g = [[1, 0, 1, 1, 0, 1, 1], [1, 1, 1, 1, 0, 0, 1]]
    b = conv(Logger().Child("t", 0), 1, 2, g, 7, [[1], [1]]).encode(rng.integers(0, 2, 40000).astype(bool))[1:]
    q = rp.symbol.load("qpsk")
    s = q.modulate(np.asarray(b[:len(b)//2*2], np.uint8))*1j
    sigma = np.sqrt(0.5*10**(-3/10))
    s = (s + sigma*(rng.standard_normal(len(s)) + 1j*rng.standard_normal(len(s)))).astype(np.complex64)
    f = F.identify(s[:2**15], "qpsk")
    assert f.found and f.kind == "conv" and f.name.endswith("/1_2")
    head = F.bit_streams(s[:2**15], q, True)
    hard, _ = M.scan_defined_conv(head)
    assert not any(h.coding[1].endswith("/1_2") and h.z > 5.6 for h in hard) # (hard alone: nothing the pick would take)

def test_product_columns_from_few_blocks():
    """A product code's column length from fewer blocks than columns the rank needs: several columns from each block"""
    from rpps.analysis.code.block import linear
    rng = np.random.default_rng(4)
    tpc = rp.code.load("block/product/tpc_64_57_sq")
    b = np.concatenate([np.asarray(tpc.encoder.encode(rng.integers(0, 2, 3249).astype(bool)), np.uint8) for _ in range(30)])
    assert linear.product(b, 64, 0, max_n2=80, per_block=1) is None # 30 blocks, one column each: too few for 64 + 24
    n2, ph, d = linear.product(b, 64, 0, max_n2=80)
    assert (n2, ph, d) == (64, 0, 7)
