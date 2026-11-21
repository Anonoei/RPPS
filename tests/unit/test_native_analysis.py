"""The C++ kernels of the blind tools give the Python reference's output (rpps._native off and on)"""
import numpy as np
import pytest

from rpps import _accel
from rpps.analysis.modulation import cpm
from rpps.analysis.code.block import ldpc
from rpps.math import gf2
from rpps import code as coding

pytestmark = pytest.mark.skipif(not _accel.available(), reason="rpps._native is not built")

class Counting:
    def __init__(self, mod):
        self.mod, self.calls = mod, 0
    def __getattr__(self, name):
        self.calls += 1
        return getattr(self.mod, name)

def both(fn):
    saved = _accel.native
    try:
        _accel.native = None
        ref = fn()
        spy = Counting(saved)
        _accel.native = spy
        got = fn()
        assert spy.calls > 0, "the native kernel never ran"
    finally:
        _accel.native = saved
    return ref, got

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
