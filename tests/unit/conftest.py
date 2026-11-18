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

def shaped(rng, points, n_sym, sps, beta=0.35, snr_db=None, esn0_db=None, cfo=0.0, tau=0.0, span=12, sym=None):
    """RRC-shaped symbols at any (fractional) samples per symbol: x(t) = sum s_k p(t - k - tau), t in symbols, sampled
    every 1/sps symbols; cfo in cycles a sample; noise at snr_db over the sample bandwidth or esn0_db per symbol.
    Returns (samples complex64, symbols sent)"""
    from rpps.signal.filters.shaping import rrc_pulse
    pts = np.asarray(points, dtype=np.complex128)
    pts = pts/np.sqrt(np.mean(np.abs(pts)**2))
    s = pts[rng.integers(0, len(pts), n_sym)] if sym is None else np.asarray(sym)
    t = np.arange(int(n_sym*sps))/sps - tau
    k0 = np.floor(t).astype(int)
    x = np.zeros(len(t), complex)
    for d in range(-span, span + 1):
        k = k0 + d
        ok = (k >= 0) & (k < n_sym)
        x[ok] += s[k[ok]]*rrc_pulse(t[ok] - k[ok], beta)
    x = x/np.sqrt(sps) # unit symbol energy: per-sample power 1/sps
    if esn0_db is not None:
        snr_db = esn0_db - 10*np.log10(sps)
    if snr_db is not None:
        var = np.mean(np.abs(x)**2)*10**(-snr_db/10)
        x = x + (rng.standard_normal(len(x)) + 1j*rng.standard_normal(len(x)))*np.sqrt(var/2)
    x = x*np.exp(2j*np.pi*cfo*np.arange(len(x)))
    return x.astype(np.complex64), s
