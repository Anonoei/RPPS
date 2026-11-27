"""The shipped example filters and windows against the formulas of the sources their "ref" names"""
import numpy as np
import pytest

import rpps as rp
from rpps.signal.filters import Window, iir

def sos(name):
    return iir.as_sos(rp.signal.filters.load(f"iir/{name}").sos())

def test_notch_is_the_audio_eq_cookbooks():
    w0, q = 2*np.pi*0.1, 30.0
    alpha = np.sin(w0)/(2*q)
    want = np.array([1, -2*np.cos(w0), 1, 1 + alpha, -2*np.cos(w0), 1 - alpha])/(1 + alpha)
    got = sos("biquad/notch_0p1_q30")[0]
    np.testing.assert_allclose(got/got[3], want, atol=1e-12)

@pytest.mark.parametrize("name,b,a", [("sos/dc_blocker_0p995", [1, -1], [1, -0.995]), # J. O. Smith: (1 - z^-1)/(1 - R z^-1)
                                      ("single_pole/alpha_0p05", [0.05, 0], [1, -0.95])]) # GNU Radio: alpha/(1 - (1 - alpha) z^-1)
def test_first_order_filters_are_their_transfer_functions(name, b, a):
    got = sos(name)
    assert len(got) == 1
    np.testing.assert_allclose(got[0]/got[0][3], b + [0] + a + [0], atol=1e-12)

def test_tukey_window_has_harris_figures():
    """Harris 1978, table I, Tukey 0.25: coherent gain 0.88, equivalent noise bandwidth 1.10 bins"""
    w = np.asarray(Window.get("tukey/a0p25")(4096))
    assert np.mean(w) == pytest.approx(0.88, abs=0.01)
    assert len(w)*np.sum(w**2)/np.sum(w)**2 == pytest.approx(1.10, abs=0.01)
