"""The C++ kernels of the blind tools give the Python reference's output (rpps._native off and on)"""
import numpy as np
import pytest

from rpps.analysis.modulation import cpm
from rpps.analysis.code.block import ldpc
from rpps.math import gf2
from rpps import code as coding
from support.native import both
from support.native import kernels

@pytest.mark.parametrize("name", ["msk", "gmsk_0.3", "soqpsk_tg"])
def test_cpm_detector(name):
    rng = np.random.default_rng(0)
    y = cpm.modulate(rng.integers(0, 2, 800), name, 8)*np.exp(0.4j)
    y = y + 0.5*(rng.standard_normal(len(y)) + 1j*rng.standard_normal(len(y)))
    (b0, m0), (b1, m1) = both(lambda: cpm.mlse(y, name, 8))
    assert np.array_equal(b0, b1) and abs(m0 - m1) < 1e-9

@pytest.mark.parametrize("shape", [(50, 70, 60), (300, 200, 150), (129, 327, 300), (700, 700, 700)])
def test_gf2_elimination(shape):
    R, C, nc = shape
    rng = np.random.default_rng(R)
    B = (rng.random((R, C)) < 0.4).astype(np.uint8)
    B[R//2:] = B[:R - R//2] ^ B[R - R//2 - 1::-1] # dependent rows
    def run():
        A = gf2.pack_rows(B)
        return gf2.eliminate_packed(A, nc), A
    (p0, a0), (p1, a1) = both(run)
    assert p0 == p1 and np.array_equal(a0, a1)

def test_flooding_min_sum():
    rng = np.random.default_rng(1)
    H = (rng.random((60, 120)) < 0.06).astype(np.uint8)
    H[0] = 0
    llr = rng.standard_normal((30, 120))*3
    t0, t1 = both(lambda: ldpc.posterior(H, llr))
    assert np.allclose(t0, t1, atol=1e-12)

@pytest.mark.parametrize("name", ["rs/204_188", "rs/ccsds_255_223"])
def test_rs_syndromes(name):
    c = coding.load(f"block/{name}").encoder
    w = np.random.default_rng(2).integers(0, 256, (500, c.den)).astype(np.int64)
    s0, s1 = both(lambda: c.syndromes(w))
    assert np.array_equal(s0, s1)

def test_rs_count_zeros_matches_python():
    """The blind RS search's sliding transform (rs._count_zeros) in C++: the same zero counts, every phase, variant, field
    and root; and a blind search through it still finds the code"""
    from rpps.analysis.code.block import rs as R
    rng = np.random.default_rng(0)
    EXP, LOG = R._tables(R.PRIMITIVE[:4])
    for n, Vn, L in ((16, 3, 400), (51, 5, 51*10)):
        V = rng.integers(0, 256, (Vn, L)).astype(np.uint8)
        with kernels(False):
            ref = R._count_zeros(V, n, EXP, LOG, 8)
        got = R._count_zeros(V, n, EXP, LOG, 8)
        assert all(np.array_equal(a, b) for a, b in zip(ref, got))

def test_find_detect():
    from rpps.analysis.spectrum import find as F
    rng = np.random.default_rng(5)
    for _ in range(60):
        n, R, M = int(rng.integers(50, 9000)), int(rng.integers(1, 5)), int(rng.choice([2, 8, 32]))
        Z = rng.gamma(M, 1/M, (R, n))
        for _ in range(rng.integers(0, 8)): # (things of every width and level)
            a, w = int(rng.integers(0, n)), int(rng.integers(1, max(2, n//4)))
            Z[:, a:a + w] *= 1 + 10**rng.uniform(-1.5, 1.5)
        det = F._Detector(8192, 8192, M, 1e-3, int(rng.choice([16, 2048])), R - 1)
        Zi = Z if R > 1 else Z[0]
        ref, got = det._detect_py(Zi, M), det.detect(Zi, M)
        assert np.array_equal(got[0], ref[0]) and np.array_equal(got[2], ref[2])
        assert np.allclose(got[1], ref[1], atol=1e-4)
