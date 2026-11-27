"""The receiver front's kernels (word and preamble metrics, carrier helpers) give the numpy results"""
import numpy as np

import rpps as rp
from support.dvbs2 import KNOWN

def test_native_word_metric_equals_the_numpy_one(monkeypatch):
    """FrameSync's header-word metric in C++ and in numpy: the same frames found, metrics equal to single precision"""
    from rpps import _accel
    p = rp.profile.load("dvbs2/normal", sync=KNOWN)
    fs = next(op for _, op in p.rx().ops if type(op).__name__ == "FrameSync")
    rng = np.random.default_rng(3)
    y = (rng.normal(size=20000) + 1j*rng.normal(size=20000)).astype(np.complex128)
    y[5000:5000 + fs.L] = fs.preamble*0.8
    c = np.convolve(y, np.conj(fs.preamble[::-1]), "valid")
    cs = np.concatenate(([0.0], np.cumsum(y.real**2 + y.imag**2)))
    m = np.abs(c)/np.sqrt(np.maximum(cs[fs.L:fs.L + len(c)] - cs[:len(c)], 1e-30)*fs._ep)
    a = fs._with_words(y, c, m, cs)
    monkeypatch.setattr(_accel, "native", None)
    b = fs._with_words(y, c, m, cs)
    np.testing.assert_allclose(a, b, rtol=1e-4, atol=1e-5)
    assert (a > 0).sum() > 100

def test_native_carrier_helpers_equal_the_numpy_ones(monkeypatch):
    from rpps import _accel
    from rpps.sync.phase import mpower
    rng = np.random.default_rng(6)
    seg = (rng.normal(size=1500) + 1j*rng.normal(size=1500))
    seg[::37] = 0.0 # zeros stay zero
    for M, phi, rate in ((8, 0.4, 3e-4), (4, -2.0, -1e-3), (2, 0.0, 0.0), (1, 1.0, 5e-3)):
        a = mpower._measure(seg, phi, rate, M)
        r = mpower._turn_back(seg, phi, rate)
        monkeypatch.setattr(_accel, "native", None)
        b = mpower._measure(seg, phi, rate, M)
        q = mpower._turn_back(seg, phi, rate)
        monkeypatch.undo()
        np.testing.assert_allclose(a[0], b[0], rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(a[1], b[1], rtol=1e-9)
        np.testing.assert_allclose(r, q, rtol=1e-5, atol=1e-6)

def test_native_preamble_metric_equals_the_numpy_one():
    from rpps import _accel
    rng = np.random.default_rng(7)
    for L in (1, 7, 26, 255):
        y = rng.normal(size=20000) + 1j*rng.normal(size=20000)
        y[3000:3000 + L] *= 4
        pre = np.exp(2j*np.pi*rng.random(L))
        c, m = _accel.native.preamble_metric(y, pre, float(np.sum(np.abs(pre)**2)))
        want_c = np.correlate(y, pre, "valid")
        cs = np.concatenate(([0.0], np.cumsum(np.abs(y)**2)))
        e = cs[L:] - cs[:-L]
        want_m = np.abs(want_c)/np.sqrt(np.maximum(e, 1e-30)*np.sum(np.abs(pre)**2))
        np.testing.assert_allclose(c, want_c, rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(m, want_m, rtol=1e-8, atol=1e-9)
