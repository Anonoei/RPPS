"""Frequency hopping: dwells, hop set, spacing, de-hopping (rpps.analysis.hopping)"""
import numpy as np
import pytest

from rpps.analysis import hopping, fsk
from conftest import cpfsk

FS = 2e6

def hopper(rng, channels, dwell_s, n_hops, snr_db=15.0):
    """GFSK-like (2-FSK, h 0.5, 50 kBd) hopping over `channels` (Hz), a dwell each"""
    n = int(dwell_s*FS)
    sps = FS/50e3
    out, seq = [], []
    for i in range(n_hops):
        c = channels[rng.integers(0, len(channels))]
        seq.append(c)
        x, _ = cpfsk(rng, int(n/sps) + 1, sps, 2, 0.5, None, snr_db=60)
        x = x[:n]*np.exp(2j*np.pi*c/FS*np.arange(n))
        out.append(x)
    x = np.concatenate(out)
    sig = 10**(-snr_db/20)
    return x + sig*(rng.standard_normal(len(x)) + 1j*rng.standard_normal(len(x)))/np.sqrt(2), seq

def test_hop_set_spacing_and_dwell():
    rng = np.random.default_rng(0)
    channels = [-600e3 + 75e3*i for i in range(17)]
    x, seq = hopper(rng, channels, 2e-3, 60)
    h = hopping.track(x, FS)
    assert h.hopping and h.dwell == pytest.approx(2e-3, rel=0.05)
    assert h.spacing == pytest.approx(75e3, rel=0.01)
    assert set(np.round(np.array(h.channels)/75e3)) <= set(np.round(np.array(channels)/75e3))
    assert len(h.channels) >= 14 # (60 random hops over 17 channels)

def test_dehopped_signal_is_one_fsk():
    rng = np.random.default_rng(1)
    x, _ = hopper(rng, [-300e3, 0.0, 150e3, 450e3], 4e-3, 30)
    h = hopping.track(x, FS)
    y = hopping.dehop(x, FS, h, guard=0.05)
    r = fsk.analyze(y, FS)
    assert r.name == "MSK" and r.baud == pytest.approx(50e3, rel=2e-3)

def test_a_steady_carrier_is_not_hopping():
    rng = np.random.default_rng(2)
    x, _ = cpfsk(rng, 4000, 40, 2, 0.5, None, snr_db=15)
    assert not hopping.track(x, FS).hopping
