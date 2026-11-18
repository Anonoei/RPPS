"""Matrix pencil lines (rpps.analysis.pencil)"""
import numpy as np
import pytest

from rpps.analysis import pencil

def test_two_lines_closer_than_a_bin():
    """256 samples at 1 MHz: an FFT bin is 3.9 kHz; tones 1.5 kHz apart (0.4 of a bin) are told apart, each to a few tens
    of Hz at -37 dB noise (exact without noise)"""
    rng = np.random.default_rng(0)
    n = np.arange(256)
    x = np.exp(2j*np.pi*10e3/1e6*n) + 0.7*np.exp(2j*np.pi*11.5e3/1e6*n + 1j)
    x = x + 0.01*(rng.standard_normal(256) + 1j*rng.standard_normal(256))
    got = pencil.lines(x, 1e6)
    f = sorted(t[0] for t in got[:2])
    assert f[0] == pytest.approx(10e3, abs=60) and f[1] == pytest.approx(11.5e3, abs=60)
    assert abs(got[0][1]) == pytest.approx(1.0, abs=0.1) and abs(got[1][1]) == pytest.approx(0.7, abs=0.1)

def test_damping():
    n = np.arange(400)
    x = np.exp((2j*np.pi*0.05 - 0.01)*n)
    f, a, d = pencil.lines(x, 1.0, K=1)[0]
    assert f == pytest.approx(0.05, abs=1e-6) and d == pytest.approx(0.01, rel=1e-3)
