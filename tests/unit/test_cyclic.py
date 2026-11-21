"""Cyclostationary lines: symbol rate, carrier order and offset, and a calibrated 'no line' (rpps.analysis.spectrum.cyclic)"""
import numpy as np
import pytest

import rpps as rp
from rpps.analysis.spectrum import cyclic
from conftest import shaped

SPS, FS, CFO = 4.37, 1e6, 0.013

@pytest.mark.parametrize("pts, beta, order", [
    (rp.symbol.PSK(2).points, 0.35, 2), (rp.symbol.PSK(4).points, 0.05, 4), (rp.symbol.PSK(8).points, 0.35, 8),
    (rp.symbol.PSK(8).points, 0.05, 8), (rp.symbol.QAM(16).points, 0.2, 4)])
def test_rate_order_and_offset(pts, beta, order):
    x, _ = shaped(np.random.default_rng(1), pts, 20000, SPS, beta=beta, esn0_db=10, cfo=CFO)
    f = cyclic.features(x, FS)
    assert f.linear and f.baud == pytest.approx(FS/SPS, rel=1e-4)
    assert f.order == order and f.cfo == pytest.approx(CFO*FS, abs=2.0)

def test_apsk_outer_ring_at_symbol_instants():
    x, _ = shaped(np.random.default_rng(2), rp.symbol.APSK(16).points, 20000, SPS, beta=0.2, esn0_db=20, cfo=CFO)
    f = cyclic.features(x, FS)
    assert f.order == 12 and f.cfo == pytest.approx(CFO*FS, abs=2.0)

def test_noise_has_no_lines():
    hits = 0
    for s in range(20):
        rng = np.random.default_rng(100 + s)
        x = ((rng.standard_normal(30000) + 1j*rng.standard_normal(30000))/np.sqrt(2)).astype(np.complex64)
        f = cyclic.features(x, FS)
        hits += f.baud is not None or f.order is not None
    assert hits <= 1 # pfa 1e-3 per test, a handful of tests a trial

def test_false_alarm_rate_is_calibrated():
    """Gaussian symbols shaped at a rate: cyclostationary in |x|^2 but circular, so no x^M line: a significant one is a
    false alarm, at about the pfa asked for"""
    tests = hits = 0
    for s in range(60):
        rng = np.random.default_rng(200 + s)
        g = (rng.standard_normal(4000) + 1j*rng.standard_normal(4000))/np.sqrt(2)
        x, _ = shaped(rng, [1], 4000, SPS, snr_db=10, sym=g)
        f = cyclic.features(x, FS, pfa=1e-2)
        assert f.linear # the rate is there
        ls = [l for l in f.lines if l.kind.startswith("x")]
        tests += len({l.kind for l in ls})
        hits += sum(l.significant for l in ls if l.kind in {m.kind for m in ls})
    assert hits <= 3*tests*1e-2 + 3

def test_constant_envelope_rate_from_the_discriminator():
    """MSK: |x|^2 is flat, so the rate comes from the instantaneous frequency"""
    rng = np.random.default_rng(4)
    sps, n = 8, 6000
    d = 2*rng.integers(0, 2, n) - 1
    ph = np.cumsum(np.repeat(d, sps))*np.pi/2/sps # h = 1/2
    x = np.exp(1j*ph) + 0.05*(rng.standard_normal(n*sps) + 1j*rng.standard_normal(n*sps))
    f = cyclic.features(x, 8e5)
    assert not f.linear and f.baud_from == "|d|^2" and f.baud == pytest.approx(1e5, rel=1e-3)
    assert f.envelope < 1.1

def test_too_short():
    f = cyclic.features(np.zeros(100, np.complex64), FS)
    assert f.baud is None and f.order is None

def test_alpha_profile_peaks_at_the_rate():
    x, _ = shaped(np.random.default_rng(3), rp.symbol.PSK(4).points, 4000, 4, esn0_db=15)
    a, p = cyclic.alpha_profile(x, 4.0)
    assert a[np.argmax(p*(a > 0.1))] == pytest.approx(1.0, abs=0.02)

def test_carrier_is_inside_the_band():
    """A tone outside the signal's band (a spur) is not its carrier; an x^M line's alias is resolved into the band"""
    rng = np.random.default_rng(6)
    x, _ = shaped(rng, rp.symbol.PSK(4).points, 20000, 8.0, beta=0.35, esn0_db=15, cfo=-0.1) # centred at -100 kHz
    n = np.arange(len(x))
    spur = 0.5*np.exp(2j*np.pi*0.35*n) # at +350 kHz, outside the band
    f = cyclic.features(x + spur, FS)
    assert f.order == 4 and f.cfo == pytest.approx(-0.1*FS, abs=5)
    assert 1 not in f.carrier
