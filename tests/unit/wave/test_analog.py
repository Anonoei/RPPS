"""Analog modes: AM, SSB and FM demodulation, and squelch"""
import numpy as np
import pytest

import rpps as rp

# --- analog ---
def test_am_recovers_the_audio():
    Fs = 48e3
    t = np.arange(48000)/Fs
    audio = 0.5*np.sin(2*np.pi*1000*t)
    x = ((1 + audio)*np.exp(2j*np.pi*5000*t)).astype(np.complex64)
    out = (rp.Data(x, Fs=Fs) + rp.wave.AMDemod()).data
    spec = np.abs(np.fft.rfft(out[8000:]))
    assert np.fft.rfftfreq(len(out) - 8000, 1/Fs)[np.argmax(spec)] == pytest.approx(1000, abs=5)
    assert np.std(out[8000:]) == pytest.approx(0.5/np.sqrt(2), rel=0.1)

@pytest.mark.parametrize("side,tone,expect_present", [("usb", 1000, True), ("usb", -1000, False), ("lsb", -1000, True), ("lsb", 1000, False)])
def test_ssb_keeps_one_sideband(side, tone, expect_present):
    Fs = 16e3
    t = np.arange(32000)/Fs
    x = np.exp(2j*np.pi*tone*t).astype(np.complex64)
    out = (rp.Data(x, Fs=Fs) + rp.wave.SSBDemod(2800, side, 201)).data[1000:]
    assert (np.sqrt(np.mean(out**2)) > 0.3) == expect_present
    assert out.dtype == np.float32

def test_fm_chain_recovers_a_tone():
    Fs, dev = 48e3, 5e3
    t = np.arange(48000)/Fs
    audio = np.sin(2*np.pi*1000*t)
    ph = 2*np.pi*dev*np.cumsum(audio)/Fs
    x = np.exp(1j*ph).astype(np.complex64)
    out = (rp.Data(x, Fs=Fs) + rp.wave.FMDemod(gain=-Fs/(2*np.pi*dev)) + rp.signal.Deemphasis(1e-9)).data[100:]
    assert np.corrcoef(out, audio[100:])[0, 1] > 0.99

def test_squelch_gates_by_level():
    rng = np.random.default_rng(0)
    quiet = 0.001*(rng.normal(size=4000) + 1j*rng.normal(size=4000))
    loud = np.exp(2j*np.pi*0.05*np.arange(4000))
    x = np.concatenate((quiet, loud, quiet)).astype(np.complex64)
    out = (rp.Data(x) + rp.sync.Squelch(-20.0)).data
    assert len(out) == len(x)
    assert np.all(out[:3000] == 0) and np.mean(np.abs(out[5000:7500])) > 0.9 and np.all(out[-2000:] == 0)

def test_fm_by_deviation_round_trips_audio():
    t = np.arange(4000)/48e3
    audio = (0.7*np.sin(2*np.pi*440*t)).astype(np.float32)
    wave = rp.Data(audio, Fs=48e3) + rp.wave.FM(deviation=5e3)
    back = rp.Data(np.asarray(wave.data), Fs=48e3) + rp.wave.FMDemod(deviation=5e3)
    assert back.DT == rp.dtype.REAL and np.allclose(np.asarray(back.data)[1:], audio[1:], atol=1e-3)

def test_am_envelope_can_keep_its_mean():
    x = np.array([0, 1, 1, 0, 1, 0, 0, 0], np.float32)*0.5 + 0j
    env = np.asarray((rp.Data(x.astype(np.complex64), Fs=1.0) + rp.wave.AMDemod(alpha=None)).data)
    assert np.allclose(env, np.abs(x))
