"""Polar frozen positions, blind (rpps.analysis.polar)"""
import numpy as np
import pytest

from rpps.analysis import polar

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
