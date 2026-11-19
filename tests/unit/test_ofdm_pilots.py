"""OFDM pilot patterns found blind (rpps.analysis.spectrum.ofdm_pilots)"""
import numpy as np

from rpps.analysis.spectrum import ofdm_pilots
from rpps.analysis.modulation.cpm import interpolate

CONT = [0, 48, 54, 87, 141, 156, 192, 201, 255, 279]

def ofdm(rng, T=240, N=512, cpf=1/4, used=301, snr=15, cfo=0.31, scattered=True, continual=True, off=None, Fs=1.37):
    """DVB-T-like: scattered pilots on every 12th subcarrier moving 3 a symbol (4 symbols a period), continual pilots,
    pilots 4/3 the data's amplitude; off: (period, on) symbols switched off (TDD). Resampled to Fs (Tu = N)"""
    ks = np.arange(used)
    k0 = -(used//2)
    pr = 1 - 2*rng.integers(0, 2, used)
    ncp = int(N*cpf)
    out = []
    for t in range(T):
        d = (rng.choice([-1, 1], used) + 1j*rng.choice([-1, 1], used))/np.sqrt(2)
        if scattered:
            sp = (ks % 12) == (3*(t % 4))
            d[sp] = 4/3*pr[sp]
        if continual:
            d[CONT] = 4/3*pr[CONT]
        if off is not None and t % off[0] >= off[1]:
            d = d*0
        X = np.zeros(N, complex)
        X[(k0 + ks) % N] = d
        s = np.fft.ifft(X)*np.sqrt(N)
        out.append(np.concatenate((s[-ncp:], s)))
    x = np.concatenate([np.zeros(137)] + out)*1.0
    x = x*np.exp(2j*np.pi*cfo/N*np.arange(len(x)))
    y = interpolate(x, np.arange(int(len(x)*Fs))/Fs)
    y = y + 10**(-snr/20)*np.sqrt(used/N/2)*(rng.standard_normal(len(y)) + 1j*rng.standard_normal(len(y)))
    return y, Fs, N, ncp

def test_scattered_and_continual_pilots():
    y, Fs, N, ncp = ofdm(np.random.default_rng(0))
    p = ofdm_pilots.discover(y, Fs, tu=N, tcp=ncp)
    assert p.period == 4 and p.spacing == 12 and p.step == 3
    assert set(CONT) <= set(p.continual.tolist())
    assert abs(p.boost_db - 20*np.log10(4/3)) < 0.5
    assert abs(p.grid.cfo*N - 0.31) < 0.02 # the fractional carrier offset, in subcarriers
    assert p.mask.shape[1] == 301

def test_no_pilots_is_said():
    y, Fs, N, ncp = ofdm(np.random.default_rng(1), scattered=False, continual=False)
    assert ofdm_pilots.discover(y, Fs, tu=N, tcp=ncp).period is None

def test_symbols_switched_off_are_left_out():
    """TDD: 3 symbols of every 8 off; the pattern is still found"""
    y, Fs, N, ncp = ofdm(np.random.default_rng(2), T=320, off=(8, 5))
    p = ofdm_pilots.discover(y, Fs, tu=N, tcp=ncp)
    assert abs(p.grid.on.mean() - 5/8) < 0.02
    assert p.spacing == 12 and p.period in (4, 8)

def test_timing_is_smooth():
    y, Fs, N, ncp = ofdm(np.random.default_rng(3))
    g = ofdm_pilots.grid(y, Fs, N, ncp)
    assert np.all(np.abs(np.diff(g.starts) - (g.N + g.ncp)) <= 1)
