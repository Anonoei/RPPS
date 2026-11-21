"""5G NR LDPC code and its rate matching (rpps.code.types.nr_ldpc) against 3GPP TS 38.212 5.2.2, 5.3.2, 5.4.2"""
import sys

import numpy as np
import pytest

import rpps as rp

nl = sys.modules["rpps.code.types.nr_ldpc"] # (the package's name nr_ldpc is the class)

def graph(bg):
    return rp._config.read(rp._config.find("fec", f"ldpc/nr_bg{bg}", "block"))["graph"]

@pytest.mark.parametrize("bg,rows,entries,first", [(1, 46, 316, 22), (2, 42, 197, 10)])
def test_base_graphs_are_the_tables(bg, rows, entries, first):
    """Tables 5.3.2-2 / 5.3.2-3: the row count, the entries, the core of four rows, and an identity ending every row past
    it (column `first` + i, V = 0 in every set)"""
    g = graph(bg)
    assert len(g) == rows and sum(map(len, g)) == entries
    for i, row in enumerate(g):
        assert all(a[0] < b[0] for a, b in zip(row, row[1:])) and all(len(e) == 9 for e in row)
        if i >= 4:
            assert row[-1] == [first + i] + [0]*8 and row[-2][0] < first + 4
    assert [len(r) for r in g[:4]] == ([19]*4 if bg == 1 else [8, 10, 8, 10])
    assert g[0][0] == ([0, 250, 307, 73, 223, 211, 294, 0, 135] if bg == 1 else [0, 9, 174, 0, 72, 3, 156, 143, 145])

@pytest.mark.parametrize("bg", [1, 2])
@pytest.mark.parametrize("Z", [s[1] for s in nl.SETS]) # One lifting size from each set
def test_codewords_satisfy_every_check(bg, Z, rng):
    ks = nl.SHAPE[bg][2]
    code = rp.code.load(f"block/ldpc/nr_bg{bg}", kp=ks*Z, e=3*ks*Z, z=Z).encoder
    assert code.Z == Z
    c = rng.integers(0, 2, (2, code.K)).astype(np.uint8)
    assert not code.syndrome(code.codeword(c).astype(bool)).any()

def test_segmentation():
    assert nl.segment(8448, 1) == (1, 8448, 8448, 384)
    assert nl.segment(8449, 1) == (2, 4249, 4576, 208) # ceil((8449 + 48)/2) bits each, 22 Z >= 4249
    assert nl.segment(100, 2) == (1, 100, 180, 18) # Kb 6: 6 Z >= 100
    assert nl.segment(3841, 2)[0] == 2
    assert [nl.base_graph(A, R) for A, R in ((292, 0.9), (3824, 0.6), (3825, 0.6), (10000, 0.2), (10000, 0.5))] == [2, 2, 1, 2, 1]

def test_rate_matching_skips_the_filler_and_starts_at_k0():
    c = rp.code.load("block/ldpc/nr_bg2", kp=150, e=2000).encoder # Kb 6: Z 26, K = 10 Z = 260: 110 filler bits
    assert (c.Z, c.K) == (26, 260)
    sent = set(c.index.tolist())
    assert not sent & set(range(150, 260)) and not sent & set(range(52)) # Filler, and the punctured first 2 Z
    assert c.index[0] == 52
    c1 = rp.code.load("block/ldpc/nr_bg2", kp=150, e=200, rv=1).encoder
    assert c1.index[0] == 52 + 13*1300//(50*26)*26 # k0 = floor(13 Ncb / 50 Z) Z, Ncb = N = 50 Z

def test_bit_interleaver_groups_by_modulation_order():
    a = rp.code.load("block/ldpc/nr_bg2", kp=150, e=400).encoder
    b = rp.code.load("block/ldpc/nr_bg2", kp=150, e=400, qm=4).encoder
    assert np.array_equal(b.index, a.index.reshape(4, 100).T.reshape(-1)) # f[i + 4j] = e[100 i + j]

@pytest.mark.parametrize("bg,kp,e,rv", [(2, 200, 600, 0), (1, 3000, 6000, 0), (2, 1000, 4000, 2), (1, 3000, 9000, 3)])
def test_decodes_through_noise(bg, kp, e, rv, rng):
    c = rp.code.load(f"block/ldpc/nr_bg{bg}", kp=kp, e=e, qm=2, rv=rv)
    m = rng.integers(0, 2, kp).astype(bool)
    x = c.encoder.encode(m)
    llr = 2*((1 - 2.0*x) + rng.normal(0, 0.6, len(x)))/0.36
    out, err = c.decoder.decode(llr, status=True)
    assert np.array_equal(out, m) and err[0] >= 0
