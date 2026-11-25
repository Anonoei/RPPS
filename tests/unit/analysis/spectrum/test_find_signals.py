"""Finding signals in a spectrum: bands, duty and measurements"""
import numpy as np
import pytest

import rpps as rp
from support.signals import two_bands

# --- dsp ------------------------------------------------------------------------------------------------

def test_dsp_exports():
    assert callable(rp.analysis.psd) and callable(rp.sync.find_baud) and callable(rp.analysis.find_signals)
    assert not hasattr(rp.utils, "psd") and hasattr(rp.utils, "arg_freq")

def test_psd_peak_at_tone():
    Fs, N = 1e6, 4096
    f = 125e3
    x = np.exp(2j*np.pi*f/Fs*np.arange(N))
    p = rp.analysis.psd(x, Fs)
    freqs = np.fft.fftshift(np.fft.fftfreq(N, 1/Fs))
    assert freqs[np.argmax(p)] == pytest.approx(f, abs=Fs/N)

def test_find_signals_locates_a_band(rng):
    Fs, N = 1e6, 1 << 18
    f = np.fft.fftfreq(N, 1/Fs)
    band = (np.abs(f - 200e3) < 30e3) # A flat 60 kHz wide band, built in the frequency domain
    spec = np.where(band, rng.normal(size=N) + 1j*rng.normal(size=N), 0)
    sig = np.fft.ifft(spec)*np.sqrt(N)*0.5
    noise = (rng.normal(size=N) + 1j*rng.normal(size=N))*0.05
    found = rp.analysis.find_signals((sig + noise).astype(np.complex64), Fs)
    assert len(found) == 1
    assert found[0].mid == pytest.approx(200e3, abs=5e3)
    assert found[0].width == pytest.approx(60e3, rel=0.2)
    assert found[0].snr > 10

def test_find_signals_in_pure_noise(rng):
    n = (rng.normal(size=1 << 17) + 1j*rng.normal(size=1 << 17)).astype(np.complex64)
    assert rp.analysis.find_signals(n, 1e6) == []

def test_find_signals_splits_close_neighbours(rng):
    Fs = 1e6
    x = two_bands(rng, 1 << 14, Fs, -120e3, 20e3, 100e3) # 40 kHz apart, edge to edge, from few samples
    found = rp.analysis.find_signals(x, Fs)
    assert len(found) == 2
    assert found[0].mid == pytest.approx(-120e3, abs=15e3) and found[1].mid == pytest.approx(20e3, abs=15e3)

def shaped(rng, N, Fs, bands, noise=0.05):
    """Noise plus flat bands [(centre, width, amplitude), ...] built in the frequency domain"""
    f = np.fft.fftfreq(N, 1/Fs)
    x = (rng.normal(size=N) + 1j*rng.normal(size=N))*noise
    for fc, bw, a in bands:
        spec = np.where(np.abs(f - fc) < bw/2, rng.normal(size=N) + 1j*rng.normal(size=N), 0)
        x = x + np.fft.ifft(spec)*np.sqrt(N)*a
    return x.astype(np.complex64)

def test_find_signals_in_a_band_mostly_full_of_signal(rng):
    Fs = 1e6 # Two transponders fill 75% of the band: the median is inside them, the noise is the gaps
    x = shaped(rng, 1 << 18, Fs, [(-230e3, 240e3, 0.3), (230e3, 240e3, 0.3)], noise=0.05)
    found = rp.analysis.find_signals(x, Fs)
    assert len(found) == 2
    assert found[0].mid == pytest.approx(-230e3, abs=15e3) and found[1].mid == pytest.approx(230e3, abs=15e3)
    assert found[0].width == pytest.approx(240e3, rel=0.15)

def test_find_signals_cut_by_band_edge(rng):
    Fs = 1e6 # Reaches into the receiver's roll-off at the top: kept, cut there, and flagged
    x = shaped(rng, 1 << 18, Fs, [(-100e3, 40e3, 0.5), (480e3, 100e3, 0.5)])
    found = rp.analysis.find_signals(x, Fs)
    assert len(found) == 2
    assert not found[0].clipped and found[1].clipped
    assert found[1].hi == pytest.approx(0.495*Fs, abs=2e3) # edge=0.5% of the band at each end
    assert "cut by band edge" in found[1].describe()

def test_duty_weak_continuous_is_not_bursty(rng):
    Fs = 1e6 # A wide signal a few dB over the noise in every bin (6 dB: processable) stays on throughout
    x = shaped(rng, 1 << 19, Fs, [(0, 200e3, 0.1)], noise=0.05)
    found = rp.analysis.find_signals(x, Fs)
    assert len(found) == 1
    assert found[0].duty is None or found[0].duty > 0.9

def test_duty_sees_bursts(rng):
    Fs, N = 1e6, 1 << 19
    x = shaped(rng, N, Fs, [(0, 100e3, 0.5)])
    clean = shaped(rng, N, Fs, [])
    on = np.repeat(np.arange(8) % 2 == 0, N//8) # Whole spectrum segments on or off
    found = rp.analysis.find_signals(np.where(on, x, clean).astype(np.complex64), Fs)
    assert len(found) == 1 and found[0].duty == pytest.approx(0.5, abs=0.15)

def shaped_psd(rng, N, Fs, shape_db, noise=1.0):
    """Complex noise whose power spectrum follows shape_db(f) [dB, f in Hz]"""
    f = np.fft.fftfreq(N, 1/Fs)
    spec = (rng.normal(size=N) + 1j*rng.normal(size=N))*10**(shape_db(f)/20)
    return (np.fft.ifft(spec)*np.sqrt(N)*noise).astype(np.complex64)

def test_a_smooth_hill_is_not_a_signal(rng):
    Fs = 100e6 # Passband curvature and bleed-over: 4 dB high, tens of MHz across, no steep edges
    x = shaped_psd(rng, 1 << 20, Fs, lambda f: 4*np.exp(-(f/15e6)**2))
    assert rp.analysis.find_signals(x, Fs) == []
    with pytest.warns(DeprecationWarning): # The old rules' thresholds: taken for a release, doing nothing
        assert rp.analysis.find_signals(x, Fs, edges=False) == []

def test_square_edged_boxes_are_signals_and_neighbours_stay_apart(rng):
    Fs = 100e6 # Two 6 dB boxes 2 MHz apart, as on a transponder pair (4.7 dB SNR: processable)
    def box(f):
        return 6.0*((np.abs(f + 15e6) < 8e6) | (np.abs(f - 3e6) < 5e6))
    found = rp.analysis.find_signals(shaped_psd(rng, 1 << 20, Fs, box), Fs)
    assert len(found) == 2
    assert found[0].mid == pytest.approx(-15e6, abs=1e6) and found[0].width == pytest.approx(16e6, rel=0.1)
    assert found[1].mid == pytest.approx(3e6, abs=1e6) and found[1].width == pytest.approx(10e6, rel=0.1)

def test_steep_block_cut_by_band_edge_is_flagged(rng):
    Fs = 100e6 # A flat top (6 dB) that runs off the end of the usable band
    found = rp.analysis.find_signals(shaped_psd(rng, 1 << 20, Fs, lambda f: 6.0*(f > 20e6)), Fs)
    assert len(found) == 1 and found[0].clipped and found[0].lo == pytest.approx(20e6, abs=1e6)

def test_signal_esn0_from_the_in_band_snr():
    s = rp.analysis.Signal(0.0, 3.0e6, -100.0, 20.0, 9.6)
    assert s.esn0_db(2.5e6) == pytest.approx(9.6 + 10*np.log10(3.0/2.5))
    assert s.esn0_db() is None # No baud known

def test_a_carrier_on_a_raised_floor_has_its_snr_over_that_floor():
    # A satellite transponder: its noise and intermodulation raise the floor 12 dB across 40 MHz, and the carriers sit on
    # it. Their SNR is over that floor (what the demodulator sees), and the raised band is marked as holding them
    import rpps as rp
    rng = np.random.default_rng(3)
    Fs, n = 100e6, 2**20
    t = np.arange(n)/Fs
    noise = lambda: (rng.standard_normal(n) + 1j*rng.standard_normal(n))/np.sqrt(2)
    lp = rp.signal.filters.low_pass_kaiser(20e6, 2e6, 60, Fs)
    ped = np.convolve(noise(), np.asarray(lp.filt), mode="same")
    ped *= np.sqrt(10**1.2/np.mean(np.abs(ped)**2)*0.4) # 12 dB over the thermal floor in its band (40 of 100 MHz)
    q = rp.symbol.load("qpsk").points
    def carrier(f, baud, cnr_db):
        sps = int(Fs/baud)
        up = np.zeros(n, complex); up[::sps] = q[rng.integers(0, 4, len(up[::sps]))]
        x = np.convolve(up, rp.signal.filters.rrc_pulse(np.arange(-8*sps, 8*sps + 1)/sps, 0.35), mode="same")
        # The pedestal's density is 10**1.2 x the thermal one; the carrier's power over it in its band is cnr
        x *= np.sqrt(10**(cnr_db/10)*10**1.2*(baud/Fs)/np.mean(np.abs(x)**2))
        return x*np.exp(2j*np.pi*f*t)
    x = noise() + ped + carrier(-10e6, 1e6, 15.0) + carrier(5e6, 1e6, 20.0)
    sv = rp.analysis.survey(x.astype(np.complex64), Fs=Fs, baud=False)
    near = lambda f: min(sv, key=lambda s: abs(s.mid - f))
    a, b = near(-10e6), near(5e6)
    assert abs(a.esn0_db(1e6) - 15.0) < 1.0 and abs(b.esn0_db(1e6) - 20.0) < 1.0 # Not 12 dB more, over the thermal floor
    big = max(sv, key=lambda s: s.width)
    assert big.width > 30e6 and big.holds >= 2
    only = rp.analysis.find_signals(x.astype(np.complex64), Fs, nested=False) # Without the pedestal: just the carriers
    assert only and not any(s.holds for s in only) and all(s.width < 30e6 for s in only)
    assert any(abs(s.mid + 10e6) < 1e6 for s in only) and any(abs(s.mid - 5e6) < 1e6 for s in only)

def test_carriers_narrower_than_the_edge_window_are_not_boxed_with_their_gaps(rng):
    # 383 kHz carriers 750 kHz apart at 100 MS/s, strong and weak by turns: narrower than the edge step's +-w (0.5% of
    # the band), so a strong one's steps are found out in its weak neighbours' skirts and box it with its gaps (1.3 MHz)
    Fs = 100e6
    mids = 6.385e6 + 750e3*np.arange(5)
    lvs = np.array([12.0, 20.0, 12.0, 20.0, 12.0])
    def row(f):
        return ((np.abs(f[:, None] - mids[None, :]) < 192e3)*lvs[None, :]).max(axis=1)
    found = rp.analysis.find_signals(shaped_psd(rng, 1 << 20, Fs, row), Fs)
    for m in mids[lvs == 20]:
        s = min(found, key=lambda s: abs(s.mid - m))
        assert s.mid == pytest.approx(m, abs=60e3) and s.width == pytest.approx(383e3, rel=0.15)
    assert all(s.width < 600e3 for s in found)

def test_a_band_given_by_hand_measures_as_a_found_one(rng):
    Fs = 1e6
    x = two_bands(rng, 1 << 17, Fs, -200e3, 150e3, 60e3)
    found = rp.analysis.find_signals(x, Fs, cf=100e6)[1]
    typed = rp.analysis.measure_band(x, Fs, 100e6 + 150e3, 60e3, cf=100e6)
    assert typed.power is not None and typed.Fs == Fs and typed.freq == pytest.approx(100e6 + 150e3, abs=1e3)
    assert typed.power == pytest.approx(found.power, abs=0.5) and typed.snr == pytest.approx(found.snr, abs=1.0)
    assert typed.duty > 0.9 and not typed.clipped
    quiet = rp.analysis.measure_band(x, Fs, 100e6 + 400e3, 60e3, cf=100e6) # Only noise there
    assert quiet.snr < typed.snr - 20
    assert rp.analysis.measure_band(x, Fs, 100e6 + 490e3, 60e3, cf=100e6).clipped
