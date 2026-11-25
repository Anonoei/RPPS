"""Blind LDPC / cyclic reconstruction: staircase solve, progressive search, roots (rpps.analysis.code.block.ldpc)"""
import numpy as np
import pytest

from rpps.analysis.code.block import ldpc as L
from rpps.analysis.code.block.linear import sparse_checks
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

def dvbs2_words(rng, table, n, k, W, Z=360):
    """Codewords of a DVB-S2 style IRA code from its address table"""
    m = n - k
    q = m//Z
    u = rng.integers(0, 2, (W, k)).astype(np.uint8)
    p = np.zeros((W, m), np.uint8)
    for g, row in enumerate(table):
        for x in row:
            rows = (x + np.arange(Z)*q) % m
            p[:, rows] ^= u[:, g*Z:(g + 1)*Z]
    p = np.cumsum(p, axis=1) % 2
    return np.concatenate((u, p.astype(np.uint8)), axis=1)

def test_qc_staircase_small_table():
    """A QC-IRA code with Z = 36: the address table back from k/Z + 12 words"""
    rng = np.random.default_rng(5)
    Z, G, q = 36, 8, 6
    n, k = (G + q)*Z, G*Z
    table = [sorted(rng.choice(q*Z, 3, replace=False).tolist()) for _ in range(G)]
    words = dvbs2_words(rng, table, n, k, 60, Z)
    r = L.qc_staircase(words, k, Z)
    assert r.table == table and np.all(r.holds == 1.0)

@pytest.mark.slow
def test_qc_staircase_dvbs2_normal_half():
    """DVB-S2 normal frame, rate 1/2 (64800, 32400): its address table from about 100 clean frames (k / 360 + 12)"""
    import json
    import rpps as rp
    path = rp._config.find("fec", "block/ldpc/dvbs2_64800_1_2")
    obj = json.load(open(path))
    rng = np.random.default_rng(6)
    words = dvbs2_words(rng, obj["table"], 64800, 32400, 110)
    r = L.qc_staircase(words, 32400, trials=1)
    assert r.table == [sorted(t) for t in obj["table"]]

def test_qc_dual_lifted_code():
    """A QC code lifted from a small base matrix (Z = 24): its check blocks back from n/Z + 8 words, and Z found"""
    rng = np.random.default_rng(7)
    Z, bc, br = 24, 8, 3 # 8 column blocks, 3 check blocks
    shifts = rng.integers(-1, Z, (br, bc)) # -1: no circulant
    shifts[:, -br:] = np.where(np.eye(br, dtype=bool), 0, -1) # an identity parity part keeps it full rank
    H = np.zeros((br*Z, bc*Z), np.uint8)
    for i in range(br):
        for j in range(bc):
            if shifts[i, j] >= 0:
                H[i*Z:(i + 1)*Z, j*Z:(j + 1)*Z] = np.roll(np.eye(Z, dtype=np.uint8), shifts[i, j], axis=1)
    from rpps.math.gf2 import nullspace, rank
    Gm = nullspace(H)
    words = ((rng.integers(0, 2, (30, len(Gm))).astype(np.int32) @ Gm.astype(np.int32)) % 2).astype(np.uint8)
    sizes = L.qc_size(words, [12, 16, 24, 48])
    assert max(sizes, key=sizes.get) == 24 and sizes[24] == br*Z
    B = L.qc_dual(words, Z)
    assert len(B) == br and all(b.sum() <= bc for b in B) # one sparse base row per block
    rows = np.concatenate([np.stack([np.roll(b.reshape(bc, Z), j, axis=1).reshape(-1) for j in range(Z)]) for b in B])
    assert not np.any((Gm.astype(np.int32) @ rows.T.astype(np.int32)) % 2) # every row is a check
    assert rank(rows) == rank(H)

# ---- quasi-cyclic: the generator's circulants from a few dozen words, sparse checks from the generator

def qc_words(rng, W):
    """802.11n's (648, 324) QC-LDPC (Z = 27), message first"""
    import rpps as rp
    enc = rp.code.load("block/ldpc/ieee80211n_648_1_2").encoder
    u = rng.integers(0, 2, (W, 324)).astype(np.uint8)
    return np.array([enc.encode(x) for x in u]).astype(np.uint8)

def test_qc_generator_and_its_sparse_checks():
    rng = np.random.default_rng(5)
    words = qc_words(rng, 80)
    llr = awgn_llr(rng, words, 11.0) # about one word in eight has an error: the reliable words solve, all verify
    noisy = (llr < 0).astype(np.uint8)
    assert (noisy != words).any()
    V, c, holds = L.qc_generator(noisy, 324, 27, rel=np.abs(llr))
    assert V.shape == (12, 12, 27) and not c.any() and np.all(holds > 0.99)
    H, r = L.qc_checks(V, 27)
    assert r == 324 == len(H) and H.sum(axis=1).max() <= 8 # 802.11n's rows are 7 and 8 ones
    assert not ((words.astype(np.int32) @ H.T.astype(np.int32)) % 2).any() # the code's checks

def test_circulant_size():
    rng = np.random.default_rng(6)
    assert L.circulant_size(qc_words(rng, 60), 324) == 27
    assert L.circulant_size(rng.integers(0, 2, (60, 648)).astype(np.uint8), 324) is None

def test_qc_checks_of_four_parity_blocks():
    """A QC-LDPC code whose block rows of checks have four parity blocks (one has three: rows of even weight alone would
    leave the parity part singular): no pair or triple of shifted parity blocks is sparse there, so those checks come
    from the quadruples (meeting in the middle)"""
    from rpps.math.gf2 import rank
    z, mb, kb = 16, 6, 6
    rng = np.random.default_rng(7)
    def circ(s):
        return np.roll(np.eye(z, dtype=np.uint8), s, axis=1)
    while True:
        Hp = np.zeros((mb*z, mb*z), np.uint8)
        Hu = np.zeros((mb*z, kb*z), np.uint8)
        for i in range(mb):
            for j in rng.choice(mb, 3 if i == 0 else 4, replace=False):
                Hp[i*z:(i+1)*z, j*z:(j+1)*z] = circ(rng.integers(z))
            for j in rng.choice(kb, 2, replace=False):
                Hu[i*z:(i+1)*z, j*z:(j+1)*z] = circ(rng.integers(z))
        if rank(Hp) == mb*z:
            break
    G, ok = L._solve(Hp, Hu) # p = Hp^-1 Hu u
    assert ok
    u = rng.integers(0, 2, (40, kb*z)).astype(np.uint8)
    words = np.concatenate((u, (u.astype(np.int32) @ G.T.astype(np.int32)) % 2), axis=1).astype(np.uint8)
    V, c, holds = L.qc_generator(words, kb*z, z)
    assert np.all(holds == 1.0)
    H, r = L.qc_checks(V, z)
    assert r == mb*z and not ((words.astype(np.int32) @ H.T.astype(np.int32)) % 2).any()
    assert H.sum(axis=1).max() <= 6

def test_sparse_checks_with_fewer_words_than_the_dimension():
    """Fewer words than k + 16: some are kept aside to verify on (before, every word went to the solve and nothing was
    returned), and sparse checks still come out of the smaller null spaces"""
    from rpps.math.gf2 import nullspace
    rng = np.random.default_rng(8)
    h = gallager(240, 3, 6, seed=1).astype(np.uint8)
    G = nullspace(h)
    words = ((rng.integers(0, 2, (100, len(G))).astype(np.int32) @ G.astype(np.int32)) % 2).astype(np.uint8)
    assert len(words) < len(G) + 16
    H, holds = sparse_checks(words, trials=200, patience=60, seed=0)
    assert len(H) and np.all(holds > 0.99) and not ((words.astype(np.int32) @ H.T.astype(np.int32)) % 2).any()

def test_qc_generator_with_constant_information_bits():
    """A header's constant bits leave the information columns short of rank: the generator still predicts the words"""
    import rpps as rp
    rng = np.random.default_rng(10)
    enc = rp.code.load("block/ldpc/ieee80211n_648_1_2").encoder
    u = rng.integers(0, 2, (60, 324)).astype(np.uint8)
    u[:, :30] = 0 # (a header that never changes)
    words = np.array([enc.encode(x) for x in u]).astype(np.uint8)
    V, c, holds = L.qc_generator(words, 324, 27)
    assert np.all(holds == 1.0)
    H, r = L.qc_checks(V, 27)
    assert r == 324 and not ((words.astype(np.int32) @ H.T.astype(np.int32)) % 2).any()
