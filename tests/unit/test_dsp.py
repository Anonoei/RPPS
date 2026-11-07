import numpy as np
import pytest

import rpps as rp
from rpps.math.interp import peak, parabolic_offset
from rpps.math.prob import mixture_variance

def qpsk_baseband(rng, nsym, sps, Fs, cfo=0.0, snr_db=20.0):
    """Half-sine shaped QPSK at Fs/sps baud, with a carrier offset and white noise"""
    sym = (rng.choice([-1, 1], nsym) + 1j*rng.choice([-1, 1], nsym))/np.sqrt(2)
    pulse = np.sin(np.pi*(np.arange(sps) + 0.5)/sps)
    x = np.convolve(np.repeat(sym, sps) if False else np.kron(sym, np.eye(1, sps)[0]), pulse)[:nsym*sps]
    x = x/np.sqrt(np.mean(np.abs(x)**2))
    n = (rng.normal(size=len(x)) + 1j*rng.normal(size=len(x)))*np.sqrt(10**(-snr_db/10)/2)
    return (x*np.exp(2j*np.pi*cfo/Fs*np.arange(len(x))) + n).astype(np.complex64)

# --- shared maths -----------------------------------------------------------------------------------

def test_parabolic_offset_recovers_vertex():
    f = lambda x: -(x - 0.3)**2
    assert parabolic_offset(f(-1), f(0), f(1)) == pytest.approx(0.3)
    assert parabolic_offset(1, 1, 1) == 0.0

@pytest.mark.parametrize("true", [10.2, 33.75, 50.5])
def test_peak_finds_sub_bin_position(true):
    x = np.arange(100)
    spec = np.exp(-(x - true)**2/(2*2.0**2))
    k, d = peak(spec)
    assert k + d == pytest.approx(true, abs=0.02)

def test_peak_on_edges():
    spec = np.array([5.0, 1.0, 1.0, 1.0])
    assert peak(spec) == (0, 0.0) # No neighbour on one side
    k, d = peak(spec, circular=True) # The other end is its neighbour
    assert k == 0 and abs(d) <= 0.5

def test_mixture_variance(rng):
    pts = np.array([1, 1j, -1, -1j])
    var = 0.04
    y = rng.choice(pts, 20000) + (rng.normal(size=20000) + 1j*rng.normal(size=20000))*np.sqrt(var)
    d2 = np.abs(y[:, None] - pts[None, :])**2
    assert mixture_variance(d2) == pytest.approx(var, rel=0.05)

# --- dsp ------------------------------------------------------------------------------------------------

def test_dsp_exports():
    assert callable(rp.dsp.psd) and callable(rp.dsp.find_baud) and callable(rp.dsp.find_signals)
    assert callable(rp.dsp.stft.stft) and callable(rp.dsp.stft.psd)
    assert not hasattr(rp.utils, "psd") and hasattr(rp.utils, "arg_freq")

@pytest.mark.parametrize("sps", [6, 8, 10])
def test_find_baud(sps, rng):
    Fs = 1e6
    x = qpsk_baseband(rng, 4000, sps, Fs, cfo=2e3)
    baud, strength = rp.dsp.find_baud(x, Fs)
    assert baud == pytest.approx(Fs/sps, rel=0.005)
    assert strength > 5
    assert rp.dsp.find_baud(x[:100], Fs) == (0.0, 0.0) # Too short to say

def test_find_baud_on_noise_is_weak(rng):
    n = (rng.normal(size=20000) + 1j*rng.normal(size=20000)).astype(np.complex64)
    assert rp.dsp.find_baud(n, 1e6)[1] < 5

def test_psd_peak_at_tone():
    Fs, N = 1e6, 4096
    f = 125e3
    x = np.exp(2j*np.pi*f/Fs*np.arange(N))
    p = rp.dsp.psd(x, Fs)
    freqs = np.fft.fftshift(np.fft.fftfreq(N, 1/Fs))
    assert freqs[np.argmax(p)] == pytest.approx(f, abs=Fs/N)

def test_stft_shapes_and_tone(rng):
    Fs, nfft = 1e5, 256
    x = np.exp(2j*np.pi*10e3/Fs*np.arange(8192)).astype(np.complex64)
    snips = rp.dsp.stft.stft(x, nfft, 0.5, "hann")
    assert snips.shape[0] == nfft and snips.shape[1] > 20
    ps = rp.dsp.stft.psd(snips, Fs)
    assert ps.shape == snips.shape
    assert (np.argmax(ps, axis=0) == np.argmax(ps[:, 0])).all() # The tone stays in one bin

def test_find_signals_locates_a_band(rng):
    Fs, N = 1e6, 1 << 18
    f = np.fft.fftfreq(N, 1/Fs)
    band = (np.abs(f - 200e3) < 30e3) # A flat 60 kHz wide band, built in the frequency domain
    spec = np.where(band, rng.normal(size=N) + 1j*rng.normal(size=N), 0)
    sig = np.fft.ifft(spec)*np.sqrt(N)*0.5
    noise = (rng.normal(size=N) + 1j*rng.normal(size=N))*0.05
    found = rp.dsp.find_signals((sig + noise).astype(np.complex64), Fs)
    assert len(found) == 1
    assert found[0].mid == pytest.approx(200e3, abs=5e3)
    assert found[0].width == pytest.approx(60e3, rel=0.2)
    assert found[0].snr > 10

def test_find_signals_in_pure_noise(rng):
    n = (rng.normal(size=1 << 17) + 1j*rng.normal(size=1 << 17)).astype(np.complex64)
    assert rp.dsp.find_signals(n, 1e6) == []
