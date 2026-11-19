"""Polar frozen positions, blind (rpps.analysis.code.block.polar)"""
import numpy as np
import pytest

from rpps.analysis.code.block import polar

def code(rng, N, frozen, W):
    G = polar.transform(N)
    u = rng.integers(0, 2, (W, N)).astype(np.uint8)
    u[:, frozen] = 0
    return (u.astype(np.int64) @ G.astype(np.int64)) % 2

def awgn(rng, x, esn0_db):
    sigma = np.sqrt(1/(2*10**(esn0_db/10)))
    y = 1 - 2.0*x + sigma*rng.standard_normal(x.shape)
    return 2*y/sigma**2

def test_frozen_set_recovered():
    """(64, 32) with the frozen set a reliability order would give (the low-weight rows first), at Es/N0 6 dB"""
    rng = np.random.default_rng(0)
    N = 64
    G = polar.transform(N)
    order = np.argsort(G.sum(axis=0), kind="stable") # column weights: light columns are the unreliable positions
    fz = np.sort(order[:32])
    llr = awgn(rng, code(rng, N, fz, 400), 6.0)
    r = polar.frozen(llr)
    assert np.array_equal(r.frozen, fz) and r.k == 32

def test_frozen_early_positions():
    """The frozen set a real code has: the unreliable synthetic channels, low row weight (early indices), whose columns of
    G are the heaviest (u_0 is the parity of every x bit)"""
    rng = np.random.default_rng(1)
    N = 64
    pc = np.array([bin(i).count("1") for i in range(N)])
    fz = np.sort(np.argsort(pc, kind="stable")[:32])
    r = polar.frozen(awgn(rng, code(rng, N, fz, 400), 6.0))
    assert np.array_equal(r.frozen, fz)

def test_transform_is_its_own_inverse():
    G = polar.transform(16).astype(np.int64)
    assert np.array_equal((G @ G) % 2, np.eye(16, dtype=np.int64))

def test_low_snr_blind_successive_cancellation():
    """(128, 64) with a polarization-weight frozen set at Es/N0 -2 dB (Eb/N0 +1 dB), 200 frames: found exactly, which
    the column test can't do at +2 dB"""
    rng = np.random.default_rng(5)
    N = 128
    fz = np.sort(polar.pw_order(N)[:64])
    llr = awgn(rng, code(rng, N, fz, 200), -2.0)
    r = polar.recognize(llr)
    assert np.array_equal(r.frozen, fz)
    s = polar.search(llr)
    assert np.array_equal(s.frozen, fz) and s.construction == "polarization weight" and s.k == 64

def test_search_corrects_a_noisy_blind_set():
    """On (256, 128), 200 frames: at -3 dB the blind set is a few positions off and the construction nearest it is
    exact; at -4 dB the construction is still named and within a few positions (0 to 4 over seeds)"""
    rng = np.random.default_rng(6)
    N = 256
    fz = np.sort(polar.pw_order(N)[:128])
    s = polar.search(awgn(rng, code(rng, N, fz, 200), -3.0))
    assert np.array_equal(s.frozen, fz)
    s = polar.search(awgn(rng, code(rng, N, fz, 200), -4.0))
    assert s.construction == "polarization weight" and len(np.setxor1d(s.frozen, fz)) <= 4

def test_sc_decodes_clean_frames():
    rng = np.random.default_rng(7)
    N = 64
    fz = np.sort(polar.pw_order(N)[:32])
    G = polar.transform(N)
    u = rng.integers(0, 2, (20, N)).astype(np.uint8)
    u[:, fz] = 0
    x = (u.astype(np.int64) @ G.astype(np.int64)) % 2
    mask = np.zeros(N, bool)
    mask[fz] = True
    got, _ = polar.sc(awgn(rng, x, 8.0), lambda i, L: np.zeros(len(L), np.uint8) if mask[i] else (L < 0).astype(np.uint8))
    assert np.array_equal(got, u)
