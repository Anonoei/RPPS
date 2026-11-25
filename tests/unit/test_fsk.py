"""FSK / MSK / GMSK parameters from the instantaneous frequency (rpps.analysis.modulation.fsk)"""
import numpy as np
import pytest

import rpps as rp
from rpps.analysis.modulation import fsk
from support.signals import cpfsk, shaped

FS = 1e6

@pytest.mark.parametrize("M, h, bt, sps, name", [
    (2, 0.5, None, 8.3, "MSK"), (2, 0.5, 0.3, 8.3, "GMSK"), (2, 1.0, None, 6.2, "2-FSK"), (4, 1/3, None, 10.7, "4-FSK"),
    (4, 0.5, 0.5, 9.0, "4-GFSK"), (8, 0.3, None, 12.1, "8-FSK")])
def test_parameters_and_symbols(M, h, bt, sps, name):
    rng = np.random.default_rng(M*10 + int(sps))
    x, a = cpfsk(rng, 4000, sps, M, h, bt, snr_db=18, cfo=0.01)
    r = fsk.analyze(x, FS)
    assert r.name == name
    assert r.baud == pytest.approx(FS/sps, rel=1e-3) and r.h == pytest.approx(h, abs=0.03)
    assert r.centre == pytest.approx(0.01*FS, abs=0.02*h*FS/sps)
    s = fsk.demodulate(x, FS, r)
    ref = (a + M - 1)//2
    agree = max(np.mean(s[i:i + 3000] == ref[j:j + 3000]) for i in range(2) for j in range(2))
    assert agree > (0.97 if bt is None and M < 8 else 0.9)

def test_linear_psk_is_not_fsk():
    x, _ = shaped(np.random.default_rng(3), rp.symbol.PSK(8).points, 4000, 6.3, beta=0.35, esn0_db=20)
    r = fsk.analyze(x, FS)
    assert r.M is None or r.envelope > 1.3 # a PSK's envelope says it is not constant even when f clusters

def test_noise_is_not_fsk():
    rng = np.random.default_rng(4)
    x = ((rng.standard_normal(40000) + 1j*rng.standard_normal(40000))/np.sqrt(2)).astype(np.complex64)
    assert fsk.analyze(x, FS).M is None

def test_demodulate_needs_fsk():
    r = fsk.analyze(np.zeros(100, np.complex64), FS)
    assert r.M is None
    with pytest.raises(ValueError):
        fsk.demodulate(np.zeros(100, np.complex64), FS, r)

@pytest.mark.parametrize("M, h", [(2, 0.5), (2, 0.7), (4, 0.25), (2, 1.0)])
def test_cpm_index_from_the_phase_lattice(M, h):
    rng = np.random.default_rng(int(10*h) + M)
    sps = 8.3
    x, _ = cpfsk(rng, 6000, sps, M, h, None, snr_db=20, cfo=0.004)
    r = fsk.analyze(x, FS)
    got, score = fsk.cpm_index(x, FS, FS/sps, r.tau if r.M else None)
    assert got == pytest.approx(h, abs=0.02) and score > 0.6
