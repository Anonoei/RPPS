"""Spectral coherence over all cycle frequencies, and trained FRESH filtering (rpps.analysis.scf)"""
import numpy as np
import pytest

import rpps as rp
from rpps.analysis import scf
from conftest import shaped

FS = 1e6

@pytest.fixture(scope="module")
def pair():
    """QPSK at 125 kBd at 0 Hz, BPSK at 100 kBd at +20 kHz, overlapping"""
    rng = np.random.default_rng(0)
    a, _ = shaped(rng, rp.symbol.PSK(4).points, 15000, 8.0, beta=0.35)
    b, _ = shaped(rng, rp.symbol.PSK(2).points, 15000, 10.0, beta=0.35)
    b = b[:len(a)]*np.exp(2j*np.pi*0.02*np.arange(len(a)))
    noise = 0.05*(rng.standard_normal(len(a)) + 1j*rng.standard_normal(len(a)))
    return a, b, noise

def near(table, alpha, conj, tol=2e3):
    return any(abs(t - alpha) < tol and c == conj for t, _, c in table)

def test_both_signals_features(pair):
    a, b, noise = pair
    t = scf.cycle_frequencies(a + 0.7*b + noise, FS)
    assert near(t, 125e3, False) and near(t, 100e3, False) # both rates
    assert near(t, 40e3, True) and near(t, 140e3, True) and near(t, -60e3, True) # the BPSK's 2 fc + k Rs
    assert not near(t, 250e3, True) # QPSK has no conjugate features

def test_noise_has_none():
    for s in range(5):
        rng = np.random.default_rng(10 + s)
        g = rng.standard_normal(120000) + 1j*rng.standard_normal(120000)
        assert scf.cycle_frequencies(g, FS) == []

def test_coherence_is_at_most_one(pair):
    a, b, noise = pair
    _, _, c = scf.coherence(b, FS)
    assert c.max() <= 1.0 + 1e-9

def test_fresh_takes_out_an_overlapping_bpsk(pair):
    """Trained on half, judged on the other: the conjugate branch at the BPSK's 2 fc beats any filter of x alone"""
    a, b, noise = pair
    y = a + b + noise
    half = len(a)//2
    sinr = lambda out: 10*np.log10(np.sum(np.abs(a[half:])**2)/np.sum(np.abs(out[half:] - a[half:])**2))
    wiener, _ = scf.fresh(y, FS, a[:half], (0.0,), taps=24, train=slice(0, half))
    fr, _ = scf.fresh(y, FS, a[:half], (-125e3, 0.0, 125e3), taps=24, conj_shifts=(40e3,), train=slice(0, half))
    assert sinr(fr) > sinr(wiener) + 1.5
