"""Blind trellis-code checks on PSK labels (rpps.analysis.tcm)"""
import numpy as np
import pytest

from rpps.analysis import tcm
from rpps.math import gf2

def noisy(rng, k, M, esn0, turn=0, conj=False):
    y = np.exp(2j*np.pi*(k + turn)/M + 0.2j)
    y = np.conj(y) if conj else y
    return y + 10**(-esn0/20)*(rng.standard_normal(len(y)) + 1j*rng.standard_normal(len(y)))/np.sqrt(2)

@pytest.mark.parametrize("states", [8, 16, 32])
def test_ungerboeck_8psk_any_phase(states):
    rng = np.random.default_rng(states)
    lab = tcm.encode(rng.integers(0, 2, (20000, 2)), list(tcm.UNGERBOECK_8PSK[states]), 3)
    y = noisy(rng, lab, 8, 9.0, turn=3, conj=states == 32)
    c = tcm.search(y, 8, max_nu=int(np.log2(states)))[0]
    assert c.match == f"Ungerboeck {states} states" and c.labels == "natural" and c.holds > 0.6

def test_pragmatic_code_on_gray_labels():
    """A rate-1/2 K5 (23, 35) code on the two low Gray label bits, the top bit uncoded: the check is the two generators
    crossed, g1 on z^0 and g0 on z^1"""
    rng = np.random.default_rng(5)
    u = rng.integers(0, 2, 20000)
    c0 = np.convolve(u, [int(b) for b in f"{0o23:05b}"])[:len(u)] % 2
    c1 = np.convolve(u, [int(b) for b in f"{0o35:05b}"])[:len(u)] % 2
    lab = c0 | c1 << 1 | rng.integers(0, 2, len(u)) << 2
    gray_to_k = np.argsort(np.arange(8) ^ (np.arange(8) >> 1))
    y = noisy(rng, gray_to_k[lab], 8, 12.0)
    c = tcm.search(y, 8, max_nu=4, labelling="gray", conj=False)[0]
    rev = lambda g: int(f"{g:05b}"[::-1], 2) # np.convolve's taps lead with delay 0; h's bit d is delay d
    assert c.labels == "gray" and c.h[2] == 0 and (c.h[0], c.h[1]) == (rev(0o35), rev(0o23))

def test_uncoded_has_no_check():
    rng = np.random.default_rng(1)
    assert tcm.search(noisy(rng, rng.integers(0, 8, 20000), 8, 15.0), 8, max_nu=4) == []

def test_fwht_matches_the_definition():
    a = np.random.default_rng(2).standard_normal(16)
    H = np.array([[(-1)**bin(i & j).count("1") for j in range(16)] for i in range(16)])
    assert np.allclose(gf2.fwht(a), H @ a)

def test_window_too_wide():
    with pytest.raises(ValueError):
        tcm.search(np.ones(100, complex), 8, max_nu=8)

def test_correlated_labels_are_not_a_code():
    """Symbols whose top label bit repeats (ISI, oversampling): a strong check that holds far less than a code's would"""
    rng = np.random.default_rng(7)
    k = rng.integers(0, 8, 20000)
    keep = rng.random(20000) < 0.55
    for i in range(1, len(k)): # The top bit is the previous symbol's 55% of the time
        if keep[i]:
            k[i] = (k[i] & 3) | (k[i - 1] & 4)
    y = noisy(rng, k, 8, 18.0)
    assert tcm.search(y, 8, max_nu=2) == []
    c = tcm.search(y, 8, max_nu=2, correlations=True)
    assert c and not c[0].code and "not a code" in str(c[0])

def test_soft_finds_the_code_hard_misses():
    """Ungerboeck 16 states at Es/N0 5 dB: hard decisions find no check, soft scoring does"""
    rng = np.random.default_rng(1)
    lab = tcm.encode(rng.integers(0, 2, (20000, 2)), list(tcm.UNGERBOECK_8PSK[16]), 3)
    y = noisy(rng, lab, 8, 5.0, turn=3)
    assert tcm.search(y, 8, max_nu=4) == []
    c = tcm.search(y, 8, max_nu=4, soft=True)
    assert c and c[0].match == "Ungerboeck 16 states"
