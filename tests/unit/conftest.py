import os
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pytest

@pytest.fixture
def rng():
    return np.random.default_rng(1234)

def bits(rng, n):
    return rng.integers(0, 2, n).astype(bool)

def awgn(syms, esn0_db, rng):
    """Add complex white noise to unit-power symbols at Es/N0. Returns (noisy, noise variance E|n|^2)"""
    var = 10**(-esn0_db/10)
    n = (rng.normal(size=len(syms)) + 1j*rng.normal(size=len(syms)))*np.sqrt(var/2)
    return syms + n.astype(np.complex64), var
