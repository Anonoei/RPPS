"""Blind interleavers: block size and shape, Forney branches, column delays (rpps.analysis.code.interleave)"""
import numpy as np

from rpps.analysis.code import interleave as IL
from rpps.code.types.interleave import rowcol

H15 = np.array([[(i + 1) >> j & 1 for i in range(15)] for j in range(4)], np.uint8) # Hamming (15, 11)

def hamming_words(rng, W):
    from rpps.math.gf2 import nullspace
    G = nullspace(H15)
    return ((rng.integers(0, 2, (W, len(G))).astype(np.int32) @ G.astype(np.int32)) % 2).astype(np.uint8)

def test_block_interleaver_size_and_shape():
    """12 Hamming words a block, written as 12 rows of 15, read by columns"""
    rng = np.random.default_rng(0)
    words = hamming_words(rng, 12*1500) # a rank test at L = 180 needs 180 + 48 rows
    N = 180
    perm = rowcol(N, 15)
    blocks = words.reshape(-1, N)[:, perm]
    bits = np.concatenate((rng.integers(0, 2, 37).astype(np.uint8), blocks.reshape(-1)))
    r = IL.block_search(bits, max_size=200, max_code=40)
    assert r.size == 180 and r.offset % 180 == 37 % 180 and r.code == 15 and (r.rows, r.cols) == (12, 15)

def test_forney_found_by_the_code_coming_back():
    rng = np.random.default_rng(1)
    u = rng.integers(0, 2, 40000)
    c0 = np.convolve(u, [1, 0, 0, 1, 1])[:len(u)] % 2
    c1 = np.convolve(u, [1, 1, 1, 0, 1])[:len(u)] % 2
    coded = np.stack((c0, c1), axis=1).reshape(-1).astype(np.uint8)
    I, M = 4, 3
    sym = coded[:len(coded)//(I*8)*(I*8)].reshape(-1, 8)
    out = np.zeros_like(sym)
    for j in range(I): # branch j delays by j M symbols
        x = sym[j::I]
        out[j::I] = np.concatenate((np.zeros((j*M, 8), np.uint8), x))[:len(x)]
    tx = out.reshape(-1)
    assert np.array_equal(IL.forney_undo(tx, I, M)[(I - 1)*I*M*8:], coded[:len(tx) - (I - 1)*I*M*8])
    r = IL.forney_search(tx, [(4, 3, 8), (4, 2, 8), (3, 3, 8), (12, 17, 8)])
    assert r[0][1] == (4, 3, 8) and r[0][0] > 3*max(t[0] for t in r if t[1] != (4, 3, 8))

def test_a_column_sent_a_row_late():
    rng = np.random.default_rng(2)
    rows, cols, F = 15, 20, 300
    frames = hamming_words(rng, F*cols).reshape(F, cols, rows).transpose(0, 2, 1) # columns are codewords
    R = frames.reshape(F*rows, cols).copy()
    R[1:, 7] = R[:-1, 7].copy() # column 7 one row late
    R[:-2, 12] = R[2:, 12].copy() # column 12 two rows early
    d, held, H = IL.column_delays(R.reshape(-1), rows, cols)
    want = np.zeros(cols, int)
    want[7], want[12] = 1, -2
    assert np.array_equal(d, want) and np.all(held > 0.99)
