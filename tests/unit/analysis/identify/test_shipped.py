"""Signals the linear chain doesn't take, named by the shipped profile whose own receive verifies on their samples
(rpps.analysis.signal_profile._framed.shipped): each fixed-rate profile on its own channel of the capture"""
import numpy as np

import rpps as rp
from rpps.analysis.signal_profile import _framed as F

def _adsb(rng, n=40):
    """Extended squitters (DF 17) of random messages from adsb/mode_s's own transmit side, a quarter reply apart"""
    tx, s, waves = rp.profile.load("adsb/mode_s").tx(), None, []
    for i in range(n):
        d = rp.Data(rng.integers(0, 2, 83).astype(bool), Fs=2e6, sps=1, stream=s, flush=i == n - 1)
        s = d.stream
        o = tx(d, fields={"df": 17})
        waves += [np.asarray(o.data), np.zeros(len(o.data)//4, np.complex64)]
    return np.concatenate(waves), o.Fs

def _capture(x, Fs, at, rng, snr_db=15.0, wide=2):
    """x moved `at` of a capture `wide` x its rate off centre, white noise over the capture"""
    n, m = len(x), wide*len(x)
    X = np.fft.fft(x)
    Z = np.zeros(m, complex)
    Z[:n//2], Z[m - (n - n//2):] = X[:n//2], X[n//2:]
    z = np.fft.ifft(Z)*wide*np.exp(2j*np.pi*at*np.arange(m))
    P = np.mean(np.abs(x[np.abs(x) > 0])**2)
    z += np.sqrt(wide*P/10**(snr_db/10)/2)*(rng.normal(size=m) + 1j*rng.normal(size=m))
    return z.astype(np.complex64), wide*Fs

def test_pulses_off_a_narrow_box_are_named_by_their_profile():
    """ADS-B replies (pulse position, 1 MHz): the survey takes the pulses' centre line for a box a few kHz wide; the
    adsb/mode_s receive on its own 2 MHz channel there verifies its CRC-24 on every reply"""
    rng = np.random.default_rng(0)
    x, Fs = _adsb(rng)
    z, Fc = _capture(x, Fs, 0.125, rng)
    got = F.shipped(z, Fc, 0.125*Fc, 2e3)
    assert got and got[0][0] == "adsb/mode_s"
    passed, failed = next(iter(got[0][2]["checks"].values()))
    assert passed >= 10 and not got[0][2]["refuted"]

def test_noise_is_no_shipped_profile():
    rng = np.random.default_rng(1)
    x = (rng.normal(size=400000) + 1j*rng.normal(size=400000)).astype(np.complex64)
    assert F.shipped(x, 4e6, 0.5e6, 0.1e6) == []

def test_the_centre_of_the_band_a_box_sits_in():
    """A band 4 MHz wide from +0.5 to +4.5 MHz, a box of a line at +1 MHz in it: its centre is +2.5 MHz (+1.5 from the
    box), within a spectrum bin or two"""
    rng = np.random.default_rng(2)
    Fs, n = 16e6, 2**18
    X = np.zeros(n, complex)
    f = np.fft.fftfreq(n, 1/Fs)
    band = (f > 0.5e6) & (f < 4.5e6)
    X[band] = rng.normal(size=band.sum()) + 1j*rng.normal(size=band.sum())
    x = np.fft.ifft(X)*np.sqrt(n) + 0.05*(rng.normal(size=n) + 1j*rng.normal(size=n))
    c, w = F._centre(x, Fs, 1e6, 9e6)
    assert abs(c - 1.5e6) < 0.1e6 and abs(w - 4e6) < 0.2e6
