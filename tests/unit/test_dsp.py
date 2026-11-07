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

def two_bands(rng, N, Fs, f1, f2, bw):
    f = np.fft.fftfreq(N, 1/Fs)
    x = 0
    for fc in (f1, f2):
        spec = np.where(np.abs(f - fc) < bw/2, rng.normal(size=N) + 1j*rng.normal(size=N), 0)
        x = x + np.fft.ifft(spec)*np.sqrt(N)*0.5
    return (x + (rng.normal(size=N) + 1j*rng.normal(size=N))*0.05).astype(np.complex64)

def test_find_signals_splits_close_neighbours(rng):
    Fs = 1e6
    x = two_bands(rng, 1 << 14, Fs, -120e3, 20e3, 100e3) # 40 kHz apart, edge to edge, from few samples
    found = rp.dsp.find_signals(x, Fs)
    assert len(found) == 2
    assert found[0].mid == pytest.approx(-120e3, abs=15e3) and found[1].mid == pytest.approx(20e3, abs=15e3)

def test_signal_fields_and_survey_pick(rng):
    Fs = 1e6
    x = two_bands(rng, 1 << 17, Fs, -200e3, 150e3, 60e3)
    sv = rp.dsp.find_signals(x, Fs, cf=100e6)
    assert isinstance(sv, list) and len(sv) == 2
    a, b = sv
    assert a.freq == pytest.approx(100e6 - 200e3, abs=5e3) and a.Fs == Fs and a.cf == 100e6
    assert a.lo < a.mid < a.hi and a.hi - a.lo == pytest.approx(a.width)
    assert a.duty > 0.9 and a.resolved
    assert sv.pick(near=140e3) is b and sv.pick(freq=100e6 - 190e3) is a and sv.pick(index=1) is b
    assert sv.pick(strongest=True) in sv
    assert a.baud == a.width # The old alias

def test_signal_channel_isolates_and_decimates(rng):
    Fs = 1e6
    x = two_bands(rng, 1 << 17, Fs, -200e3, 150e3, 60e3)
    sv = rp.dsp.find_signals(x, Fs)
    ch = sv[1].channel()
    assert ch.decimation > 1 and ch.Fs == pytest.approx(Fs/ch.decimation)
    out = (rp.Data(x, Fs=Fs) + ch.front).data
    f, p = np.fft.fftshift(np.fft.fftfreq(len(out), 1/ch.Fs)), np.fft.fftshift(np.abs(np.fft.fft(out))**2)
    assert abs(f[np.argmax(p)]) < 40e3 # Brought to baseband
    assert p[np.abs(f) < 35e3].sum() > 50*p[np.abs(f) > 90e3].sum() # The neighbour is gone

def test_survey_from_a_file_keeps_its_place(rng, tmp_path):
    Fs = 1e6
    x = two_bands(rng, 1 << 17, Fs, -200e3, 150e3, 60e3)
    path = tmp_path / "cap.cf64"
    x.tofile(path)
    src = rp.serial.File("cf64", str(path), Fs=Fs, cf=50e6)
    src.seek(1000)
    sv = rp.dsp.survey(src, samples=1 << 16)
    assert src.cur_samp == 1000 # Peeking doesn't use samples up
    assert len(sv) == 2 and len(sv.head) == 1 << 16 and sv[0].cf == 50e6

def test_choose_without_a_display_takes_the_strongest(rng, capsys):
    x = two_bands(rng, 1 << 16, 1e6, -200e3, 150e3, 60e3)
    sv = rp.dsp.find_signals(x, 1e6)
    import matplotlib
    matplotlib.use("Agg")
    assert sv.choose() is sv.pick(strongest=True)
    assert "strongest" in capsys.readouterr().out

def test_signal_esn0_from_the_in_band_snr():
    s = rp.dsp.Signal(0.0, 3.0e6, -100.0, 20.0, 9.6)
    assert s.esn0_db(2.5e6) == pytest.approx(9.6 + 10*np.log10(3.0/2.5))
    assert s.esn0_db() is None # No baud known
