"""Peak interpolation and mixture variance (rpps.math)"""
import numpy as np
import pytest

from rpps.math.interp import peak, parabolic_offset
from rpps.math.prob import mixture_variance

# --- shared maths -----------------------------------------------------------------------------------

def test_parabolic_offset_recovers_vertex():
    f = lambda x: -(x - 0.3)**2
    assert parabolic_offset(f(-1), f(0), f(1)) == pytest.approx(0.3)
    assert parabolic_offset(1, 1, 1) == 0.0

@pytest.mark.parametrize("true", [10.2, 33.75, 50.5])
def test_peak_finds_sub_bin_position(true):
    x = np.arange(100)
    spec = np.exp(-(x - true)**2/(2*2.0**2))
    k, d = peak(spec)
    assert k + d == pytest.approx(true, abs=0.02)

def test_peak_on_edges():
    spec = np.array([5.0, 1.0, 1.0, 1.0])
    assert peak(spec) == (0, 0.0) # No neighbour on one side
    k, d = peak(spec, circular=True) # The other end is its neighbour
    assert k == 0 and abs(d) <= 0.5

def test_mixture_variance(rng):
    pts = np.array([1, 1j, -1, -1j])
    var = 0.04
    y = rng.choice(pts, 20000) + (rng.normal(size=20000) + 1j*rng.normal(size=20000))*np.sqrt(var)
    d2 = np.abs(y[:, None] - pts[None, :])**2
    assert mixture_variance(d2) == pytest.approx(var, rel=0.05)
