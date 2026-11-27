"""Bit sequences filtered over GF(2) with no loop over the bits (rpps.math.gf2.fir, recur) against the recursions"""
import numpy as np
import pytest

from rpps.math import gf2

def loop(u, delays):
    y = np.zeros(len(u), np.uint8)
    for n in range(len(u)):
        v = int(u[n])
        for d in delays:
            if n - d >= 0:
                v ^= int(y[n - d])
        y[n] = v
    return y

@pytest.mark.parametrize("delays", [[1], [1, 2], [14, 17], [4, 7], [1, 23], [3, 3, 5], [18, 23], [2, 9, 11], []])
@pytest.mark.parametrize("n", [0, 1, 5, 300, 20011])
def test_recur_is_the_recursion(delays, n, rng):
    u = rng.integers(0, 2, n).astype(np.uint8)
    want = loop(u, delays) # (a delay twice cancels: over GF(2) it is there an even number of times)
    assert np.array_equal(gf2.recur(u, delays), want)
    for chunk in (1, 7, 64):
        assert np.array_equal(gf2.recur(u, delays, chunk=chunk), want)

def test_fir_undoes_recur(rng):
    u = rng.integers(0, 2, 5000).astype(np.uint8)
    for delays in ([1, 2], [14, 17], [3, 20]):
        assert np.array_equal(gf2.fir(gf2.recur(u, delays), delays), u)

def test_rank_by_the_native_elimination_is_the_reference_loops():
    """gf2.rank (the C++ elimination's pivots) against its Python loop on products of known rank, every shape"""
    rng = np.random.default_rng(0)
    for R, n, k in ((10, 20, 5), (200, 300, 150), (64, 64, 64), (500, 128, 90), (1, 1, 1), (30, 700, 30)):
        for _ in range(4):
            M = (rng.integers(0, 2, (R, k)) @ rng.integers(0, 2, (k, n)) % 2).astype(np.uint8)
            assert gf2.rank(M) == gf2._rank_python(M) <= k
