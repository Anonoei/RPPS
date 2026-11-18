"""Compression, IQ imbalance, phase noise, neighbours (rpps.analysis.impairments)"""
import numpy as np
import pytest

import rpps as rp
from rpps.analysis import impairments as I
from conftest import shaped

def noisy(rng, pts, esn0, n=20000):
    pts = np.asarray(pts)/np.sqrt(np.mean(np.abs(np.asarray(pts))**2))
    s = pts[rng.integers(0, len(pts), n)]
    return s + 10**(-esn0/20)*(rng.standard_normal(n) + 1j*rng.standard_normal(n))/np.sqrt(2), pts

def test_amplifier_compression_and_am_pm():
    rng = np.random.default_rng(0)
    y, pts = noisy(rng, rp.symbol.APSK(16).points, 25)
    r = np.abs(y)**2
    y = y/(1 + 0.15*r)*np.exp(1j*0.12*r) # Saleh-like: gain down and phase up with power
    a = I.amplifier(y, pts)
    assert len(a.rings) == 2 and a.compression_db < -1.0 and a.am_pm > 5.0
    clean = I.amplifier(noisy(rng, rp.symbol.APSK(16).points, 25)[0], pts)
    assert abs(clean.compression_db) < 0.1 and abs(clean.am_pm) < 0.5

def test_iq_imbalance_measured_and_corrected():
    rng = np.random.default_rng(1)
    s, pts = noisy(rng, rp.symbol.QAM(16).points, 30, 50000)
    k = 0.05*np.exp(0.3j)
    y = s + k*np.conj(s) + 0.02
    q = I.iq(y)
    assert q.ratio == pytest.approx(k, abs=0.005) and q.irr_db == pytest.approx(-20*np.log10(0.05), abs=1.0)
    assert abs(q.dc) == pytest.approx(0.02, abs=0.005)
    assert I.iq(q.corrected).irr_db > 40

def test_phase_noise_excess_and_slowness():
    rng = np.random.default_rng(2)
    y, pts = noisy(rng, rp.symbol.PSK(4).points, 20, 40000)
    p0 = I.phase_noise(y, pts)
    assert p0.excess_deg < 0.6*p0.white_deg
    walk = np.cumsum(rng.standard_normal(len(y))*0.01)
    walk = walk - np.convolve(walk, np.ones(2001)/2001, mode="same") # what a carrier loop leaves: slow wander
    p1 = I.phase_noise(y*np.exp(1j*walk), pts)
    assert p1.excess_deg > p1.white_deg and p1.slow > 0.3

def test_neighbours_touching_or_apart():
    rng = np.random.default_rng(3)
    a, _ = shaped(rng, rp.symbol.PSK(4).points, 20000, 8.0, beta=0.35, snr_db=25)
    b, _ = shaped(rng, rp.symbol.PSK(4).points, 20000, 8.0, beta=0.35)
    n = np.arange(len(a))
    near = I.neighbours(a + 0.3*b*np.exp(2j*np.pi*0.14*n), 1e6) # 140 kHz apart, 125 kBd x 1.35: overlapping
    far = I.neighbours(a + 0.3*b*np.exp(2j*np.pi*0.3*n), 1e6)
    # (the overlap itself, where both add, may come out as a band of its own)
    assert any(b.touching and b.lo < 140e3 < b.hi for b in near)
    assert len(far) == 1 and not far[0].touching
