"""Regenerate-and-subtract: how much of a signal we are locked to, and what is under it (rpps.analysis.cancel)"""
import numpy as np
import pytest

import rpps as rp
from rpps.analysis import cancel, cyclic
from conftest import shaped

SPS, FS = 4.37, 1e6

@pytest.fixture(scope="module")
def qpsk():
    rng = np.random.default_rng(0)
    return shaped(rng, rp.symbol.PSK(4).points, 12000, SPS, beta=0.25, esn0_db=15, tau=0.3, cfo=2e-4)

def test_locked_leaves_only_noise(qpsk):
    x, s = qpsk
    c = cancel.cancel(x, FS, s, SPS, tau=0.3*SPS)
    assert abs(c.excess_db) < 0.4 and abs(c.loss_db) < 0.4 and c.cfo == pytest.approx(200, abs=5)
    assert np.all(c.profile < 2)

def test_phase_noise_shows_as_excess(qpsk):
    x, s = qpsk
    pn = np.cumsum(np.random.default_rng(1).standard_normal(len(x))*0.01)
    c = cancel.cancel(x*np.exp(1j*pn), FS, s, SPS, tau=0.3*SPS)
    assert c.excess_db > 1.0

def test_wrong_decisions_show_where_they_are(qpsk):
    """(A slip, every symbol turned alike, is taken up by the block fits: it shows only at its ends)"""
    x, s = qpsk
    s2 = s.copy()
    wrong = 6000 + np.flatnonzero(np.random.default_rng(3).random(1000) < 0.3) # a burst of wrong decisions
    s2[wrong] *= 1j
    c = cancel.cancel(x, FS, s2, SPS, tau=0.3*SPS)
    bad = np.flatnonzero(c.profile > 3)
    block = int(256*SPS) # samples a fit (and a profile entry) covers
    assert len(bad) and np.all((bad + 1)*block > 6000*SPS) and np.all(bad*block < 7000*SPS)

def test_a_weak_signal_underneath_is_found(qpsk):
    x, s = qpsk
    w, _ = shaped(np.random.default_rng(2), rp.symbol.PSK(2).points, 20000, 3.1, beta=0.35, cfo=0.03)
    y = x + 0.2*w[:len(x)] # 14 dB under, overlapping the QPSK's band
    assert cyclic.features(y, FS).order == 4 # the strong one hides it
    c = cancel.cancel(y, FS, s, SPS, tau=0.3*SPS)
    f = cyclic.features(c.residual, FS)
    assert f.baud == pytest.approx(FS/3.1, rel=1e-3) and f.order == 2 and f.cfo == pytest.approx(0.03*FS, abs=5)

def test_align_finds_the_symbols(qpsk):
    x, s = qpsk
    y = np.concatenate((np.zeros(777, np.complex64), x))
    t = cancel.align(y, s, SPS)
    assert abs(t - (777 + 0.3*SPS)) < SPS/2
    assert abs(cancel.cancel(y, FS, s, SPS, tau=t).excess_db) < 0.4

def test_decide():
    pts = rp.symbol.PSK(4).points
    assert np.allclose(cancel.decide(np.asarray(pts)*1.1, pts), pts)
