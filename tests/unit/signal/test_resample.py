"""resample_poly"""
import numpy as np
import pytest

import rpps as rp

@pytest.fixture
def rng():
    return np.random.default_rng(7)

# ---- resample_poly

RATES = [(1, 5), (2, 3), (3, 2), (48, 125), (7, 1), (147, 160), (1, 625)]

@pytest.mark.parametrize("up,down", RATES)
def test_resample_poly_length_and_tone(up, down, rng):
    n = 20000 + 80*max(up, down) # long enough that the middle is clear of the filter edges
    f = 0.2*min(1, up/down) # a tone well inside the lower rate's band
    x = np.exp(2j*np.pi*f*np.arange(n))
    y = rp.signal.sample.resample_poly(x, up, down)
    assert len(y) == -(-n*up//down)
    m = np.arange(len(y))
    mid = slice(len(y)//4, 3*len(y)//4) # clear of the edges
    assert np.allclose(y[mid], np.exp(2j*np.pi*f*m*down/up)[mid], atol=2e-3) # zero delay: output m is the input at m*down/up

def test_resample_poly_dtypes_and_identity(rng):
    x = rng.standard_normal(1000).astype(np.float32)
    assert rp.signal.sample.resample_poly(x, 1, 4).dtype == np.float32
    assert rp.signal.sample.resample_poly(x.astype(np.complex64), 1, 4).dtype == np.complex64
    assert rp.signal.sample.resample_poly(x.astype(np.float64), 1, 4).dtype == np.float64
    assert np.array_equal(rp.signal.sample.resample_poly(x, 3, 3), x)
    assert len(rp.signal.sample.resample_poly(x[:0], 1, 4)) == 0
    with pytest.raises(ValueError):
        rp.signal.sample.resample_poly(x, 0, 4)

@pytest.mark.oracle
@pytest.mark.parametrize("up,down", RATES)
@pytest.mark.parametrize("cplx", [False, True])
def test_resample_poly_matches_scipy(up, down, cplx, rng):
    ss = pytest.importorskip("scipy.signal")
    x = rng.standard_normal(6001) + (1j*rng.standard_normal(6001) if cplx else 0)
    assert np.allclose(rp.signal.sample.resample_poly(x, up, down), ss.resample_poly(x, up, down), atol=1e-6)
    assert np.allclose(rp.signal.sample.resample_poly(x, up, down, window=("kaiser", 8.0)), ss.resample_poly(x, up, down, window=("kaiser", 8.0)), atol=1e-6)
