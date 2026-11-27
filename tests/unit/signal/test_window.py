"""Window functions (rpps.signal.filters.Window): every built-in, and the stage that weights blocks by one"""
import numpy as np
import pytest

import rpps as rp
from rpps.signal.filters import Window

NAMES = Window.s()

@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("N", [64, 65])
def test_window_is_a_bounded_symmetric_taper(name, N):
    w = np.asarray(Window.get(name)(N), float)
    assert w.shape == (N,) and np.all(np.isfinite(w))
    assert w.max() <= 1 + 1e-6 and w.max() > 0.98 and w.min() > -0.08 # (flat-top dips below zero)
    symmetric = np.max(np.abs(w - w[::-1]))
    periodic = np.max(np.abs(w[1:] - w[1:][::-1])) # as an FFT window: symmetric about sample N/2
    assert min(symmetric, periodic) < 1e-9
    assert np.mean(w[N//4:-N//4]) >= np.mean(w) - 1e-9 # weight in the middle, not at the ends

@pytest.mark.parametrize("name", ["tukey", "planck_taper"])
def test_a_window_built_from_halves_takes_any_length(name):
    for N in (1, 2, 3, 7, 33):
        w = Window.get(name)(N)
        assert len(w) == N and np.array_equal(w, w[::-1])
        assert N > 1 or w[0] == 1

@pytest.mark.parametrize("name,numpy", [("hann", np.hanning), ("hamming_orig", np.hamming), ("blackman", np.blackman),
                                        ("bartlett", np.bartlett)])
def test_numpys_windows_are_numpys(name, numpy):
    assert np.array_equal(Window.get(name)(33), numpy(33))

@pytest.mark.parametrize("N", [None, 32])
def test_the_stage_weights_each_block(N):
    x = np.ones(32, np.complex64)
    out = rp.Data(x, Fs=1.0) + Window("hann", N)
    np.testing.assert_allclose(np.asarray(out.data).real, np.hanning(32), atol=1e-6)

def test_a_window_of_a_set_length_refuses_another():
    with pytest.raises(ValueError, match="given a block of 16"):
        rp.Data(np.ones(16, np.complex64), Fs=1.0) + Window("hann", 32)
