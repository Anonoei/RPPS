"""Channel impairments: noise, carrier offset, fading and clock drift"""
import numpy as np
import pytest

import rpps as rp

# --- channel ---
@pytest.mark.parametrize("db", [0, 6, 15])
def test_awgn_sets_the_snr(db, rng):
    x = np.exp(2j*np.pi*0.01*np.arange(100000)).astype(np.complex64)
    y = (rp.Data(x, Fs=1e5) + rp.signal.channel.AWGN(db, seed=3)).data
    noise = y - x
    assert 10*np.log10(1/np.mean(np.abs(noise)**2)) == pytest.approx(db, abs=0.15)

def test_awgn_esn0_uses_the_sps():
    x = np.ones(200000, dtype=np.complex64)
    y = (rp.Data(x, sps=4) + rp.signal.channel.AWGN(esn0_db=10, power=1.0, seed=1)).data
    assert 10*np.log10(1/np.mean(np.abs(y - x)**2)) == pytest.approx(10 - 10*np.log10(4), abs=0.15)

def test_awgn_is_repeatable_and_seeds_differ():
    x = np.ones(1000, dtype=np.complex64)
    a, b, c = ((rp.Data(x) + rp.signal.channel.AWGN(5, seed=s)).data for s in (1, 1, 2))
    assert np.array_equal(a, b) and not np.array_equal(a, c)

def test_cfo_leaves_the_centre_frequency():
    d = rp.Data(np.ones(100, dtype=np.complex64), Fs=1e4, cf=1e6) + rp.signal.channel.CFO(300)
    assert d.cf == 1e6
    f = np.angle(d.data[1:]*np.conj(d.data[:-1]))*1e4/(2*np.pi)
    assert np.allclose(f, 300, atol=1e-2)

def test_fading_has_unit_mean_power_and_the_right_spread():
    x = np.ones(200000, dtype=np.complex64)
    g = (rp.Data(x, Fs=1e4) + rp.signal.channel.Fading(100, seed=5)).data
    assert np.mean(np.abs(g)**2) == pytest.approx(1.0, rel=0.25)
    spec = np.abs(np.fft.fftshift(np.fft.fft(g - g.mean())))**2
    f = np.fft.fftshift(np.fft.fftfreq(len(g), 1e-4))
    inside = spec[np.abs(f) <= 110].sum()/spec.sum()
    assert inside > 0.97 # The spectrum is confined to +-fd

def test_rician_fading_is_less_deep_than_rayleigh():
    x = np.ones(50000, dtype=np.complex64)
    ray = np.abs((rp.Data(x, Fs=1e4) + rp.signal.channel.Fading(50, seed=1)).data)
    ric = np.abs((rp.Data(x, Fs=1e4) + rp.signal.channel.Fading(50, k_db=10, seed=1)).data)
    assert np.std(ric) < np.std(ray)

def test_clock_drift_walks_the_timing():
    ramp = np.arange(100000, dtype=np.float32)
    y = (rp.Data(ramp) + rp.signal.channel.ClockDrift(1000)).data # Reads 1000 ppm faster than the clock
    assert y[-1] == pytest.approx(ramp[len(y)-1]*1.001, rel=1e-4)
