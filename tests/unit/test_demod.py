"""rp.wave.demod: one block of analog signal to audio, with its measurements"""
import numpy as np
import pytest

import rpps as rp

Fs = 240e3
t = np.arange(48000)/Fs
rng = np.random.default_rng(0)
noise = lambda s: s*(rng.standard_normal(len(t)) + 1j*rng.standard_normal(len(t)))

def test_fm_tone_deviation_offset_and_sinad():
    x = np.exp(1j*(2*np.pi*3e3*t + 5.0*np.sin(2*np.pi*1e3*t))) + noise(0.02)  # beta 5 at 1 kHz
    audio, aFs, m = rp.wave.demod(x, Fs, "fm", bw=40e3)
    assert m.offset_hz == pytest.approx(3e3, abs=50)
    assert m.deviation_hz == pytest.approx(5e3, rel=0.05)
    assert m.deviation_rms_hz == pytest.approx(5e3/np.sqrt(2), rel=0.05)
    assert m.tone_hz == pytest.approx(1e3, abs=20) and m.sinad_db > 20
    assert aFs <= Fs and aFs >= 2*15e3 and len(audio) == int(np.ceil(len(t)/(Fs/aFs)))

def test_pm_phase_deviation_and_offset():
    x = np.exp(1j*(2*np.pi*2e3*t + 0.8*np.sin(2*np.pi*1.5e3*t) + 0.3)) + noise(0.01) # 0.8 rad at 1.5 kHz, 2 kHz off
    audio, aFs, m = rp.wave.demod(x, Fs, "pm")
    assert m.offset_hz == pytest.approx(2e3, abs=5) and m.phase_dev_rad == pytest.approx(0.8, rel=0.05)
    assert m.phase_dev_rms_rad == pytest.approx(0.8/np.sqrt(2), rel=0.05) and m.tone_hz == pytest.approx(1.5e3, abs=20)
    assert np.max(np.abs(audio)) == pytest.approx(0.8, rel=0.1) # Radians

def test_am_depth():
    x = (1 + 0.5*np.cos(2*np.pi*800*t)) + noise(0.005)
    audio, aFs, m = rp.wave.demod(x, Fs, "am")
    assert m.depth == pytest.approx(0.5, abs=0.03) and m.tone_hz == pytest.approx(800, abs=20)
    assert np.max(audio) == pytest.approx(0.5, abs=0.05) # Audio in units of the carrier

def test_ssb_keeps_its_side():
    x = np.exp(2j*np.pi*1.2e3*t) + noise(0.001) # A tone 1.2 kHz above the carrier: upper sideband
    u, _, mu = rp.wave.demod(x, Fs, "usb")
    l, _, ml = rp.wave.demod(x, Fs, "lsb")
    assert mu.tone_hz == pytest.approx(1.2e3, abs=20) and mu.power_db > ml.power_db + 30

def test_noise_has_no_tone_and_bad_kind_is_refused():
    _, _, m = rp.wave.demod(noise(1.0), Fs, "fm")
    assert m.tone_hz is None and m.sinad_db is None
    with pytest.raises(ValueError, match="kind"):
        rp.wave.demod(noise(1.0), Fs, "qam")

@pytest.mark.parametrize("kind", ["fm", "pm", "am", "usb"])
def test_single_precision_and_the_whole_span(kind):
    """complex64 is worked in single precision with the same measurements; bw = Fs and audio_bw = Fs/2 filter nothing
    (the whole span: the same as a filter just wider than the signal)"""
    x = {"fm": np.exp(1j*5.0*np.sin(2*np.pi*1e3*t)), "pm": np.exp(1j*0.8*np.sin(2*np.pi*1.5e3*t)),
         "am": 1 + 0.5*np.cos(2*np.pi*800*t), "usb": np.exp(2j*np.pi*1.2e3*t)}[kind] + noise(0.001)
    a64, _, m64 = rp.wave.demod(x.astype(np.complex64), Fs, kind, bw=Fs, audio_bw=Fs/2)
    a, _, m = rp.wave.demod(x, Fs, kind, bw=Fs, audio_bw=Fs/2)
    assert a64.dtype == np.float32 and np.allclose(a64, a, atol=1e-4*np.max(np.abs(a)))
    for k in m.__slots__[1:]:
        if getattr(m, k) is not None:
            assert getattr(m64, k) == pytest.approx(getattr(m, k), rel=1e-3, abs=1e-3)
    if kind != "usb": # (the whole span: the audio straight from the channel, decimated)
        wide = rp.wave.demod(x, Fs, kind, bw=0.99*Fs, audio_bw=0.49*Fs)[2]
        assert m.tone_hz == wide.tone_hz
