"""Blind LDPC / cyclic reconstruction: staircase solve, progressive search, roots (rpps.analysis.ldpc)"""
import numpy as np
import pytest

from rpps.analysis import ldpc as L
from rpps.analysis.block import sparse_checks
from rpps.code.types.ldpc import gallager

def ira(rng, k, m, w=4):
    """An IRA code: A (m x k) with w ones a row, parity accumulated"""
    A = np.zeros((m, k), np.uint8)
    for j in range(m):
        A[j, rng.choice(k, w, replace=False)] = 1
    return A

def ira_words(rng, A, W):
    m, k = A.shape
    u = rng.integers(0, 2, (W, k)).astype(np.uint8)
    d = (u.astype(np.int32) @ A.T.astype(np.int32)) % 2
    p = np.cumsum(d, axis=1) % 2
    return np.concatenate((u, p.astype(np.uint8)), axis=1)

def awgn_llr(rng, words, ebn0_db, rate=0.5):
    """BPSK through AWGN: LLRs 2y/sigma^2 (errors are the bits with small |LLR|)"""
    sigma = np.sqrt(1/(2*rate*10**(ebn0_db/10)))
    y = 1 - 2.0*words + sigma*rng.standard_normal(words.shape)
    return 2*y/sigma**2

def test_staircase_rows_exactly():
    rng = np.random.default_rng(0)
    A = ira(rng, 96, 96)
    words = ira_words(rng, A, 600)
    llr = awgn_llr(rng, words, 10.0) # ~0.07% raw bit errors (rate 1/2: Es/N0 7 dB)
    H, holds = L.staircase((llr < 0).astype(np.uint8), 96, rel=np.abs(llr))
    assert np.array_equal(H[:, :96], A) and np.all(holds > 0.9)

def test_staircase_needs_information_first():
    rng = np.random.default_rng(1)
    words = np.zeros((300, 40), np.uint8)
    with pytest.raises(ValueError):
        L.staircase(words, 20)

def test_progressive_finds_more_than_one_search():
    """A (120, 62) Gallager code at Eb/N0 6.5 dB (1.7% raw errors): one search finds a few checks, the progressive search
    the whole dual (rank 58)"""
    from rpps.math.gf2 import nullspace, rank
    rng = np.random.default_rng(2)
    h = gallager(120, 3, 6, seed=3).astype(np.uint8)
    G = nullspace(h) # the code: words are combinations of its basis
    words = (rng.integers(0, 2, (1500, len(G))).astype(np.int32) @ G.astype(np.int32)) % 2
    for eb in (7, 6.5): # (the generator's draws as the measurement was made)
        llr = awgn_llr(rng, words.astype(np.uint8), eb)
    once, _ = sparse_checks((llr < 0).astype(np.uint8), rel=np.abs(llr), seed=0)
    H, holds, hist = L.progressive(llr)
    assert (rank(once) if len(once) else 0) < 10 and hist[-1] == 120 - len(G) == rank(H)
    assert not np.any((G.astype(np.int32) @ H.T.astype(np.int32)) % 2) # every check found is the code's

def test_bch_roots():
    """BCH(15, 7), g = x^8 + x^7 + x^6 + x^4 + 1: roots alpha^1..4 and their conjugates; designed distance 5"""
    rng = np.random.default_rng(3)
    g = 0b111010001
    gb = np.array([g >> i & 1 for i in range(9)])
    msg = rng.integers(0, 2, (2000, 7))
    words = np.array([np.convolve(u, gb) % 2 for u in msg]) # position i: x^i
    words ^= (rng.random(words.shape) < 0.02) # 26% of words have an error or more
    r = L.roots(words)
    assert r.generator == g and r.k == 7 and r.designed == 5 and r.roots == [1, 2, 3, 4, 6, 8, 9, 12]

def test_random_words_have_no_roots():
    r = L.roots(np.random.default_rng(4).integers(0, 2, (2000, 31)))
    assert r.roots == [] and "no roots" in str(r)
