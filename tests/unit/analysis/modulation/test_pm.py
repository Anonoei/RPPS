"""PM with a residual carrier: carrier, index, data on the carrier, subcarriers (rpps.analysis.modulation.pm)"""
import numpy as np
import pytest

from rpps.analysis.modulation import pm
from rpps.analysis.spectrum import cyclic

FS = 1e6

def nrz(rng, n, sps):
    d = 2*rng.integers(0, 2, int(n/sps) + 2) - 1
    return d[(np.arange(n)/sps).astype(int)].astype(float)

def channel(rng, phase, fc=-1200.0, snr_db=20.0):
    n = np.arange(len(phase))
    x = np.exp(1j*(phase + 2*np.pi*fc/FS*n + 0.3))
    return x + 10**(-snr_db/20)*(rng.standard_normal(len(x)) + 1j*rng.standard_normal(len(x)))/np.sqrt(2)

def test_nrz_on_the_carrier():
    rng = np.random.default_rng(0)
    m = 0.9 # rad
    p = pm.analyze(channel(rng, m*nrz(rng, 400000, 1e6/50e3)), FS)
    assert p.carrier == pytest.approx(-1200, abs=2) and p.index == pytest.approx(m, abs=0.05)
    assert p.rate == pytest.approx(50e3, rel=2e-3) and not p.biphase and not p.subcarriers
    y = pm.nrz(p.phase, FS, p.rate, p.tau)
    assert np.mean(np.abs(y) > 0.5*m) > 0.95 # bits sit at +-m

def test_bpsk_on_a_subcarrier_is_not_a_symbol_rate():
    """PCM/PSK/PM: BPSK at 8 kBd on a 64 kHz sine subcarrier. phi^2 (and |x|^2) have a line at 128 kHz, the subcarrier
    doubled, which a rate search on |x|^2 takes for a symbol rate; it is a subcarrier, with its own rate"""
    rng = np.random.default_rng(1)
    n = 400000
    t = np.arange(n)/FS
    m = 1.0
    x = channel(rng, m*nrz(rng, n, FS/8e3)*np.sin(2*np.pi*64e3*t))
    p = pm.analyze(x, FS)
    assert any(abs(f - 64e3) < 50 for f, _ in p.subcarriers) and p.rate is None
    s = pm.subcarrier(p.phase, FS, 64e3, 30e3)
    f = cyclic.features(s, FS, lo=1e3)
    assert f.baud == pytest.approx(8e3, rel=2e-3) and 2 in f.carrier # BPSK on it

def test_no_carrier_is_not_pm():
    rng = np.random.default_rng(2)
    x = (rng.standard_normal(100000) + 1j*rng.standard_normal(100000))/np.sqrt(2)
    assert pm.analyze(x, FS).carrier is None

def test_drifting_carrier_beside_a_spur():
    """A long capture: the carrier drifts 15 Hz (spread over many fine bins) and a steady spur stands taller; the carrier
    is the line that holds the power when tracked"""
    rng = np.random.default_rng(3)
    n = 2**22
    t = np.arange(n)/FS
    drift = 2*np.pi*np.cumsum(-1200 - 15*t/t[-1])/FS
    x = np.exp(1j*(0.8*nrz(rng, n, FS/20e3) + drift))
    x = x + 0.2*np.exp(2j*np.pi*-29e3*t) + 0.05*(rng.standard_normal(n) + 1j*rng.standard_normal(n))
    p = pm.analyze(x, FS)
    assert p.carrier == pytest.approx(-1200, abs=20) and p.index == pytest.approx(0.8, abs=0.08)

def test_rate_lines_without_data_are_not_data():
    """Tones near DC (no data on the carrier) give phi^2 lines; their soft symbols are not two clusters"""
    rng = np.random.default_rng(4)
    n = 400000
    t = np.arange(n)/FS
    x = channel(rng, 0.5*np.sin(2*np.pi*3e3*t) + 0.4*np.sin(2*np.pi*5e3*t))
    assert pm.analyze(x, FS).rate is None
