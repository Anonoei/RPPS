"""The C++ demapper against the array one"""
import numpy as np
import pytest

import rpps as rp
from support.signals import named, noisy_symbols

# ---- the C++ demapper against the array one

@pytest.fixture
def numpy_demap(monkeypatch):
    """Both implementations of rpps.math.demap on the same input: (native, numpy)"""
    from rpps import _accel
    from rpps.math import demap
    def both_(y, c, **kw):
        lab = c.labels().astype(bool)
        a = demap.llr_rows(np, y, c.points, lab, demap.mask_matrix(lab), **kw)
        monkeypatch.setattr(_accel, "native", None)
        b = demap.llr_rows(np, y, c.points, lab, demap.mask_matrix(lab), **kw)
        monkeypatch.undo()
        return a, b
    return both_

@pytest.mark.parametrize("name,noise", [("QPSK", 0.2), ("8PSK", 0.1), ("16QAM", 0.06), ("psk/8psk dvb-s2", 0.1), ("32APSK", 0.05)])
@pytest.mark.parametrize("kw", [{}, {"maxlog": True}, {"noise_var": 0.05}])
def test_native_demap_equals_numpy(numpy_demap, name, noise, kw):
    c = named(name)
    a, b = numpy_demap(noisy_symbols(c, 500, 3, noise, 11).astype(np.complex128), c, **kw)
    assert a.shape == b.shape and a.dtype == b.dtype == np.float32
    np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-4)

@pytest.mark.parametrize("name", ["psk/qpsk dvb-s2", "8PSK", "32APSK"])
def test_native_demap_of_long_rows_equals_numpy(numpy_demap, name):
    """Rows longer than EM_SYMBOLS (the noise from every em_stride-th symbol) and than a C++ task (split over threads)"""
    c = named(name)
    a, b = numpy_demap(noisy_symbols(c, 10007, 2, 0.08, 12).astype(np.complex128), c)
    np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-4)

def test_native_demap_where_the_sums_underflow_and_per_row_noise(numpy_demap):
    c = rp.symbol.load("8PSK")
    rng = np.random.default_rng(1)
    s = c.points[rng.integers(0, 8, (3, 2000))]
    s = (s + 1e-5*(rng.normal(size=s.shape) + 1j*rng.normal(size=s.shape)))
    s[0, ::97] *= 3
    a, b = numpy_demap(s, c)
    assert np.isfinite(a).all()
    np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-3)
    a, b = numpy_demap(s, c, noise_var=np.array([0.01, 0.1, 1.0]))
    np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-3)

def test_native_demap_empty_and_single_symbol(numpy_demap):
    c = rp.symbol.load("QPSK")
    a, b = numpy_demap(np.zeros((2, 0), np.complex128), c)
    assert a.shape == b.shape == (2, 0)
    a, b = numpy_demap(np.array([[0.3 + 0.2j]]), c)
    np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-4)
