"""Equalizers, analog demodulation, OFDM"""
import numpy as np
import pytest

import rpps as rp

def qpsk_symbols(rng, n):
    return (np.exp(1j*(np.pi/4 + np.pi/2*rng.integers(0, 4, n)))).astype(np.complex64)

# --- equalizers ---
def test_cma_opens_the_eye_through_multipath(rng):
    s = qpsk_symbols(rng, 30000)
    chan = rp.channel.Multipath([1.0, 0.45, 0.25j], [0, 1, 2])
    x = rp.Data(s, sps=1) + chan + rp.channel.AWGN(30, seed=1)
    ref = np.abs(x.data[100:2000])
    out = (x + rp.equalize.CMA(9, 0.002)).data
    # Modulus spread: before, the ISI makes |x| scatter; after, it sits near 1
    assert np.std(np.abs(out[-3000:])) < 0.5*np.std(ref)

def test_lms_with_training_converges(rng):
    s = qpsk_symbols(rng, 6000)
    x = rp.Data(s, sps=1) + rp.channel.Multipath([1.0, 0.4], [0, 1])
    eq = rp.equalize.LMS(7, 0.02, points=[1/np.sqrt(2)*(a + 1j*b) for a in (1, -1) for b in (1, -1)])
    out = (x + eq).data
    pts = np.array([a + 1j*b for a in (1, -1) for b in (1, -1)])/np.sqrt(2)
    tail = out[-1000:]
    err = np.min(np.abs(tail[:, None] - pts[None, :]), axis=1)
    assert np.mean(err) < 0.2

def test_fractionally_spaced_equalizer_halves_the_rate(rng):
    x = rp.Data((rng.normal(size=2000) + 1j*rng.normal(size=2000)).astype(np.complex64), Fs=2e4, sps=2)
    out = x + rp.equalize.CMA(9, 0.001)
    assert len(out) == 1000 and out.Fs == 1e4 and out.sps == 1

# --- analog ---
def test_am_recovers_the_audio():
    Fs = 48e3
    t = np.arange(48000)/Fs
    audio = 0.5*np.sin(2*np.pi*1000*t)
    x = ((1 + audio)*np.exp(2j*np.pi*5000*t)).astype(np.complex64)
    out = (rp.Data(x, Fs=Fs) + rp.demod.AMDemod()).data
    spec = np.abs(np.fft.rfft(out[8000:]))
    assert np.fft.rfftfreq(len(out) - 8000, 1/Fs)[np.argmax(spec)] == pytest.approx(1000, abs=5)
    assert np.std(out[8000:]) == pytest.approx(0.5/np.sqrt(2), rel=0.1)

@pytest.mark.parametrize("side,tone,expect_present", [("usb", 1000, True), ("usb", -1000, False), ("lsb", -1000, True), ("lsb", 1000, False)])
def test_ssb_keeps_one_sideband(side, tone, expect_present):
    Fs = 16e3
    t = np.arange(32000)/Fs
    x = np.exp(2j*np.pi*tone*t).astype(np.complex64)
    out = (rp.Data(x, Fs=Fs) + rp.demod.SSBDemod(2800, side, 201)).data[1000:]
    assert (np.sqrt(np.mean(out**2)) > 0.3) == expect_present
    assert out.dtype == np.float32

def test_fm_chain_recovers_a_tone():
    Fs, dev = 48e3, 5e3
    t = np.arange(48000)/Fs
    audio = np.sin(2*np.pi*1000*t)
    ph = 2*np.pi*dev*np.cumsum(audio)/Fs
    x = np.exp(1j*ph).astype(np.complex64)
    out = (rp.Data(x, Fs=Fs) + rp.mod.FMDemod(gain=-Fs/(2*np.pi*dev)) + rp.demod.Deemphasis(1e-9)).data[100:]
    assert np.corrcoef(out, audio[100:])[0, 1] > 0.99

def test_squelch_gates_by_level():
    rng = np.random.default_rng(0)
    quiet = 0.001*(rng.normal(size=4000) + 1j*rng.normal(size=4000))
    loud = np.exp(2j*np.pi*0.05*np.arange(4000))
    x = np.concatenate((quiet, loud, quiet)).astype(np.complex64)
    out = (rp.Data(x) + rp.demod.Squelch(-20.0)).data
    assert len(out) == len(x)
    assert np.all(out[:3000] == 0) and np.mean(np.abs(out[5000:7500])) > 0.9 and np.all(out[-2000:] == 0)

def test_dc_block_removes_the_offset():
    x = (0.7 + 0.1*np.exp(2j*np.pi*0.1*np.arange(20000))).astype(np.complex64)
    out = (rp.Data(x) + rp.demod.DCBlock()).data
    assert abs(np.mean(out[10000:])) < 1e-2

# --- OFDM ---
def bits_for_ofdm(rng, mod, n_ofdm, ofdm):
    return rng.integers(0, 2, mod.constellation.bits_per_symbol*ofdm.data_per_symbol*n_ofdm).astype(bool)

def test_ofdm_round_trip(rng):
    ofdm, qam = rp.mod.OFDM(), rp.mod.QAM(16)
    bits = bits_for_ofdm(rng, qam, 20, ofdm)
    wave = (rp.Data(bits) * qam) * ofdm
    assert len(wave) == 20*80
    assert np.array_equal(((wave / ofdm) / qam).bits(), bits)

def test_ofdm_equalizes_mild_multipath_and_a_phase_offset(rng):
    ofdm, qam = rp.mod.OFDM(), rp.mod.QAM(16)
    bits = bits_for_ofdm(rng, qam, 60, ofdm)
    wave = (rp.Data(bits) * qam) * ofdm
    rx = wave + rp.channel.Multipath([1.0, 0.3j, -0.15], [0, 1, 2]) + rp.channel.PhaseOffset(0.8) + rp.channel.AWGN(35, power=None, seed=2)
    assert np.mean(((rx / ofdm) / qam).bits() != bits) < 0.01

def test_ofdm_with_more_pilots_handles_a_longer_delay_spread(rng):
    # The channel changes faster across carriers the longer its echoes: the pilots have to sample it often enough
    pilots = tuple(k for k in range(-24, 25, 3) if k != 0)
    ofdm, qam = rp.mod.OFDM(pilots=pilots), rp.mod.QAM(16)
    bits = bits_for_ofdm(rng, qam, 60, ofdm)
    wave = (rp.Data(bits) * qam) * ofdm
    rx = wave + rp.channel.Multipath([1.0, 0.5j, -0.3], [0, 3, 9]) + rp.channel.AWGN(35, power=None, seed=2)
    assert np.mean(((rx / ofdm) / qam).bits() != bits) < 0.02

def test_ofdm_streams_in_chunks(rng):
    ofdm, qam = rp.mod.OFDM(), rp.mod.QAM(4)
    bits = bits_for_ofdm(rng, qam, 12, ofdm)
    syms = (rp.Data(bits) * qam).data
    whole = (rp.Data(syms, flush=True) * ofdm).data
    ctx, outs = None, []
    for a, b in ((0, 100), (100, 101), (101, 300), (300, len(syms))):
        out = rp.Data(syms[a:b], ctx=ctx) * ofdm
        ctx = out.ctx
        outs.append(out.data)
    assert np.allclose(np.concatenate(outs), whole, atol=1e-6)

def test_lms_takes_the_detected_constellation(rng):
    s = qpsk_symbols(rng, 6000)
    x = rp.Data(s, sps=1) + rp.channel.Multipath([1.0, 0.4], [0, 1])
    eq = rp.equalize.LMS(7, 0.02) # No points: they come from the detection
    assert len(x + eq) == 0 # Nothing detected yet: nothing to decide against, nothing out
    x.stream.det = type("Det", (), {"mod": "qpsk"})()
    out = (x + eq).data
    pts = np.array([a + 1j*b for a in (1, -1) for b in (1, -1)])/np.sqrt(2)
    assert np.mean(np.min(np.abs(out[-1000:, None] - pts[None, :]), axis=1)) < 0.2
