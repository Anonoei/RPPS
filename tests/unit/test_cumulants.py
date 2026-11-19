"""Higher-order cumulants: theory values, blind classification (rpps.analysis.modulation.cumulants)"""
import numpy as np
import pytest

from rpps.analysis.modulation import cumulants as C

def noisy(rng, pts, esn0, n=20000, phase=0.7):
    pts = np.asarray(pts)/np.sqrt(np.mean(np.abs(pts)**2))
    s = pts[rng.integers(0, len(pts), n)]*np.exp(1j*phase)
    return s + 10**(-esn0/20)*(rng.standard_normal(n) + 1j*rng.standard_normal(n))/np.sqrt(2)

@pytest.mark.parametrize("name, c40, c42, c63", [
    ("BPSK", 2, -2, 16), ("QPSK", 1, -1, 4), ("8PSK", 0, -1, 4), ("16QAM", 0.68, -0.68, 2.08), ("64QAM", 0.619, -0.619, 1.797)])
def test_swami_sadler_values(name, c40, c42, c63):
    v = C._vector(C.theory(C.candidates()[name]))
    assert v[0] == pytest.approx(c40, abs=2e-3) and v[1] == pytest.approx(c42, abs=2e-3)
    assert v[2] == pytest.approx(c63, abs=2e-3)

def test_gaussian_noise_has_none():
    rng = np.random.default_rng(0)
    g = (rng.standard_normal(200000) + 1j*rng.standard_normal(200000))/np.sqrt(2)
    c = C.measure(g)
    assert all(abs(v) < 0.05 for k, v in c.items() if k in ("C40", "C42"))

@pytest.mark.parametrize("name", ["BPSK", "QPSK", "8PSK", "16PSK", "32QAM", "16APSK", "32APSK"])
def test_families_blind(name):
    """No SNR given, any carrier phase: the noise share is fitted"""
    r = C.classify(noisy(np.random.default_rng(1), C.candidates()[name], 12.0))
    assert r[0].name == name and r[0].esn0_db == pytest.approx(12, abs=1.5)

@pytest.mark.parametrize("name", ["16QAM", "64QAM", "256QAM", "4ASK", "8ASK"])
def test_orders_with_the_snr(name):
    """(64 and 256QAM differ by 0.014 in C42: the scatter of 20000 symbols' estimate; 256QAM gets 200000)"""
    n = 200000 if name == "256QAM" else 20000
    r = C.classify(noisy(np.random.default_rng(2), C.candidates()[name], 15.0, n), esn0_db=15.0)
    assert r[0].name == name

def test_partitions_count():
    assert [len(C._partitions(p)) for p in range(1, 9)] == [1, 2, 5, 15, 52, 203, 877, 4140] # Bell numbers
