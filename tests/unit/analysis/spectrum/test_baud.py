"""Blind symbol rate: find_baud and its spurs"""
import numpy as np
import pytest

import rpps as rp

def qpsk_baseband(rng, nsym, sps, Fs, cfo=0.0, snr_db=20.0):
    """Half-sine shaped QPSK at Fs/sps baud, with a carrier offset and white noise"""
    sym = (rng.choice([-1, 1], nsym) + 1j*rng.choice([-1, 1], nsym))/np.sqrt(2)
    pulse = np.sin(np.pi*(np.arange(sps) + 0.5)/sps)
    x = np.convolve(np.repeat(sym, sps) if False else np.kron(sym, np.eye(1, sps)[0]), pulse)[:nsym*sps]
    x = x/np.sqrt(np.mean(np.abs(x)**2))
    n = (rng.normal(size=len(x)) + 1j*rng.normal(size=len(x)))*np.sqrt(10**(-snr_db/10)/2)
    return (x*np.exp(2j*np.pi*cfo/Fs*np.arange(len(x))) + n).astype(np.complex64)

@pytest.mark.parametrize("sps", [6, 8, 10])
def test_find_baud(sps, rng):
    Fs = 1e6
    x = qpsk_baseband(rng, 4000, sps, Fs, cfo=2e3)
    baud, strength = rp.sync.find_baud(x, Fs)
    assert baud == pytest.approx(Fs/sps, rel=0.005)
    assert strength > 5
    assert rp.sync.find_baud(x[:100], Fs) == (0.0, 0.0) # Too short to say

def test_find_baud_on_noise_is_weak(rng):
    n = (rng.normal(size=20000) + 1j*rng.normal(size=20000)).astype(np.complex64)
    assert rp.sync.find_baud(n, 1e6)[1] < 5

def test_reject_spur_bauds():
    from rpps.analysis.spectrum.signal import Signal, reject_spur_bauds
    def sig(width, baud):
        s = Signal(0.0, width, 0.0, 0.0)
        s.baud_est = baud
        return s
    sigs = [sig(1.0e6, 683e3), sig(1.6e6, 683e3), sig(1.9e6, 683e3), # One rate on very different widths: a spur
            sig(1.3e6, 1366e3), sig(1.2e6, 341.5e3), # Its harmonic and subharmonic
            sig(1.2e6, 1.0e6), sig(1.2e6, 1.0e6), sig(1.2e6, 1.0e6), # Carriers of one service: kept
            sig(100e3, 500e3)] # Faster than it is wide
    reject_spur_bauds(sigs)
    assert [s.baud_est for s in sigs] == [None]*5 + [1.0e6]*3 + [None]
