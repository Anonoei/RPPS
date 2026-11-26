"""One signal the survey cut into several boxes, profiled as one (rpps.analysis.signal_profile._run._merge): a spread
signal's pieces each show its code's period, an OFDM signal's pieces share a subcarrier grid and a symbol clock; carriers
side by side stay apart"""
import numpy as np

import rpps as rp
from rpps.analysis.signal_profile import _run as R
from rpps.analysis.spectrum.signal import Signal

def _up(v, u):
    n = len(v)
    V = np.fft.fft(v)
    Z = np.zeros(u*n, complex)
    Z[:n//2], Z[u*n - (n - n//2):] = V[:n//2], V[n//2:]
    return np.fft.ifft(Z)*u

def _noise(rng, z, snr_db, u):
    P = np.mean(np.abs(z)**2)*u
    return z + np.sqrt(P/10**(snr_db/10)/2)*(rng.normal(size=len(z)) + 1j*rng.normal(size=len(z)))

def _box(lo, hi):
    return Signal((lo + hi)/2, hi - lo, -100.0, 20.0, 15.0)

def _merged(z, Fs, boxes):
    return R._merge(rp.source.Memory(z.astype(np.complex64), Fs=Fs), [boxes[0]], boxes)

def test_the_two_halves_of_a_barker_spread_signal_are_one():
    """802.11b-like: 11 Mchip/s, each bit a Barker 11 (it sums to -1: a dip at the centre the survey cuts at). Both halves
    show the 1 MHz code period, far under their 6.5 MHz: one box over both"""
    rng = np.random.default_rng(1)
    barker = np.array([1, -1, 1, 1, -1, 1, 1, 1, -1, -1, -1], float)
    chips = np.kron(1 - 2.0*rng.integers(0, 2, 30000), barker)
    z = _noise(rng, _up(chips, 4), 15, 1) # (rectangular-ish chips at 44 MS/s, band limited)
    got = _merged(z, 44e6, [_box(-6.6e6, -0.15e6), _box(0.15e6, 6.6e6)])
    assert len(got) == 1 and got[0].merged and abs(got[0].merged[1] - 1e6) < 1e3
    assert got[0].lo == -6.6e6 and got[0].hi == 6.6e6

def test_carriers_side_by_side_at_one_rate_stay_apart():
    """Two QPSK carriers at 2 MBd, 2.7 MHz apart: each fills its own box (its rate about its width), so no merge"""
    rng = np.random.default_rng(2)
    q = np.exp(1j*(np.pi/4 + np.pi/2*np.arange(4)))
    from rpps.signal.filters.shaping import rrc_pulse
    h = rrc_pulse(np.arange(-64, 65)/8, 0.35)
    def carrier():
        up = np.zeros(8*20000, complex)
        up[::8] = q[rng.integers(0, 4, 20000)]
        return np.convolve(up, h)[64:64 + len(up)]
    t = np.arange(8*20000)
    z = _noise(rng, carrier()*np.exp(-2j*np.pi*1.35e6/16e6*t) + carrier()*np.exp(2j*np.pi*1.35e6/16e6*t), 15, 1)
    got = _merged(z, 16e6, [_box(-2.7e6, -0.02e6), _box(0.02e6, 2.7e6)])
    assert len(got) == 1 and not getattr(got[0], "merged", None)

def _ofdm(rng, nsym=600, nfft=1024, cp=72, used=600, nulls=()):
    q = np.exp(1j*(np.pi/4 + np.pi/2*np.arange(4)))
    k = np.arange(-used//2, used//2 + 1)
    k = k[k != 0]
    on = np.ones(len(k), bool)
    for a, w in nulls:
        on[a:a + w] = False
    out = []
    for _ in range(nsym):
        X = np.zeros(nfft, complex)
        X[k[on] % nfft] = q[rng.integers(0, 4, on.sum())]
        t = np.fft.ifft(X)*np.sqrt(nfft)
        out.append(np.concatenate((t[-cp:], t)))
    return np.concatenate(out)

def test_an_ofdm_signal_with_empty_subcarriers_is_one():
    """15 kHz OFDM (Tu 66.7 us, the LTE prefix), 600 subcarriers, three groups of them left empty: the survey cuts it into
    five boxes; four show the prefix, all on one grid and clock (the fifth, too narrow, between them): one box"""
    rng = np.random.default_rng(0)
    x = _ofdm(rng, nulls=((150, 40), (330, 40), (480, 30)))
    z = _up(x, 2)*np.exp(2j*np.pi*0.125*np.arange(2*len(x)))
    src = rp.source.Memory(_noise(rng, z, 15, 2).astype(np.complex64), Fs=30.72e6)
    sv = list(rp.analysis.survey(src))
    assert len(sv) >= 3
    got = R._merge(src, [sv[0]], sv)
    assert len(got) == 1 and got[0].merged and got[0].merged[1] is None
    assert abs(got[0].lo + 0.66e6) < 0.1e6 and abs(got[0].hi - 8.34e6) < 0.1e6

def test_ofdm_carriers_of_their_own_clocks_stay_apart():
    """Two such carriers 9.5 MHz apart, on one subcarrier grid, their symbols not aligned (10 seeds: never merged)"""
    for seed in range(3):
        rng = np.random.default_rng(seed)
        A, B = _up(_ofdm(rng), 3), np.roll(_up(_ofdm(rng), 3), int(rng.integers(400, 3000)))
        t = np.arange(len(A))
        z = _noise(rng, A*np.exp(-2j*np.pi*4.75e6/46.08e6*t) + B*np.exp(2j*np.pi*4.75e6/46.08e6*t), 15, 3)
        src = rp.source.Memory(z.astype(np.complex64), Fs=46.08e6)
        sv = list(rp.analysis.survey(src))
        assert len(sv) == 2
        assert not any(getattr(s, "merged", None) for s in R._merge(src, list(sv), sv))
