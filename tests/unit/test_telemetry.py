"""PM telemetry end to end: subcarrier rate, BPSK, line code, idle, frames and counters (rpps.analysis.telemetry)"""
import numpy as np
import pytest

from rpps.analysis import telemetry as T

FS = 1e6

def frames(rng, n_frames, P=256, sync=0xEB90A5, fixed=64):
    """n_frames frames of P bits: a 24-bit sync, a 16-bit counter, `fixed` more fixed bits, the rest random"""
    head = np.array([int(c) for c in format(sync, "024b")], np.uint8)
    pad = rng.integers(0, 2, fixed).astype(np.uint8)
    out = []
    for i in range(n_frames):
        cnt = np.array([int(c) for c in format((1000 + i) % 65536, "016b")], np.uint8)
        out.append(np.concatenate((head, cnt, pad, rng.integers(0, 2, P - 40 - fixed).astype(np.uint8))))
    return np.concatenate(out)

def differential(bits):
    return (np.cumsum(bits) % 2).astype(np.uint8)

def manchester(bits):
    return np.stack((bits, 1 - bits), axis=1).ravel()

def pm_signal(rng, chips, baud, fsc, m=0.8, fc=-1200.0, snr_db=25.0):
    n = int(len(chips)*FS/baud)
    t = np.arange(n)/FS
    d = 2.0*chips[(t*baud).astype(int)] - 1
    phase = m*d*np.sin(2*np.pi*fsc*t) # BPSK on a sine subcarrier (a square one sampled at a non-integer rate aliases)
    x = np.exp(1j*(phase + 2*np.pi*fc*t + 0.4))
    return x + 10**(-snr_db/20)*(rng.standard_normal(n) + 1j*rng.standard_normal(n))/np.sqrt(2)

def test_differential_frames_on_a_subcarrier():
    rng = np.random.default_rng(3)
    bits = frames(rng, 24)
    x = pm_signal(rng, differential(bits), 4096.0, 65536.0)
    t = T.analyze(x, FS)
    assert t is not None and t.pm.carrier == pytest.approx(-1200, abs=3)
    s = next(s for s in t.subcarriers if abs(s.freq - 65536) < 100)
    assert s.ok and s.baud == pytest.approx(4096, rel=2e-3) and not s.manchester and s.differential
    assert s.frame is not None and s.frame.period == 256 and s.frame.fixed >= 80
    assert any(w == 16 and step == 1 for _, w, step in s.frame.counters)

def test_manchester_halves_the_rate():
    rng = np.random.default_rng(4)
    bits = frames(rng, 20, P=320)
    x = pm_signal(rng, manchester(bits), 9600.0, 48000.0)
    s = next(s for s in T.analyze(x, FS).subcarriers if abs(s.freq - 48000) < 100)
    assert s.ok and s.manchester and s.rate == pytest.approx(4800, rel=2e-3)
    assert s.frame is not None and s.frame.period == 320

def test_idle_subcarrier():
    rng = np.random.default_rng(5)
    chips = np.tile(np.array([1, 0], np.uint8), 1500)
    s = next(s for s in T.analyze(pm_signal(rng, chips, 1000.0, 16000.0), FS).subcarriers if abs(s.freq - 16000) < 50)
    assert s.ok and s.baud == pytest.approx(1000, rel=2e-3) and s.idle == "alternating 1010"

def test_no_residual_carrier_is_none():
    rng = np.random.default_rng(6)
    q = (2*rng.integers(0, 2, 200000) - 1 + 1j*(2*rng.integers(0, 2, 200000) - 1))/np.sqrt(2)
    assert T.analyze(np.repeat(q, 4), FS) is None

def test_twice_the_rate_comes_in_equal_pairs():
    rng = np.random.default_rng(7)
    b = rng.integers(0, 2, 2000).astype(np.uint8)
    assert T.doubled(np.repeat(b, 2)) and not T.doubled(b)

def test_frame_period_is_the_frame_not_a_multiple():
    rng = np.random.default_rng(8)
    L, r, f = T.period(frames(rng, 40))
    assert L == 256 and f > 0.3
    assert T.period(rng.integers(0, 2, 10000).astype(np.uint8))[0] is None

def test_phase_modulation_mirrors_about_its_carrier_and_a_tone_beside_a_carrier_does_not():
    rng = np.random.default_rng(5)
    x = pm_signal(rng, rng.integers(0, 2, 600).astype(np.uint8), 4096.0, 65536.0, snr_db=15.0)
    sym, side = T.symmetry(x, FS, -1200.0)
    assert sym > T.SYMMETRIC and side > T.SIDEBANDS
    sps = 20 # A QPSK carrier at +100 kHz and a tone at -60 kHz
    n = sps*2**15
    sym = (rng.choice([-1, 1], n//sps) + 1j*rng.choice([-1, 1], n//sps))/np.sqrt(2)
    t = np.arange(n)
    y = np.repeat(sym, sps)*np.exp(2j*np.pi*100e3/FS*t) + 0.8*np.exp(-2j*np.pi*60e3/FS*t)
    y = y + 0.05*(rng.standard_normal(n) + 1j*rng.standard_normal(n))
    assert T.symmetry(y, FS, -60e3)[0] < T.SYMMETRIC
    tone = np.exp(2j*np.pi*3e3/FS*t) + 0.1*(rng.standard_normal(n) + 1j*rng.standard_normal(n)) # A tone in noise
    assert T.symmetry(tone, FS, 3e3)[1] < T.SIDEBANDS

def test_a_tone_in_band_limited_noise_has_no_sidebands():
    # A channel filtered to its band: the noise is flat inside and down in the stopband, so against the median bin it
    # would stand clear on both sides of a centred tone, and mirror
    from scipy.signal import firwin, lfilter
    rng = np.random.default_rng(7)
    n, band = 2**19, 300e3
    noise = lfilter(firwin(255, band/2, fs=FS), 1, rng.standard_normal(n) + 1j*rng.standard_normal(n))
    noise *= np.sqrt(0.5/np.mean(np.abs(noise)**2))
    x = np.exp(2j*np.pi*2e3/FS*np.arange(n)) + noise # 3 dB SNR in the band
    assert T.symmetry(x, FS, 2e3)[1] > T.SIDEBANDS # The median bin is the stopband's
    assert T.symmetry(x, FS, 2e3, band=band, snr_db=3.0)[1] < T.SIDEBANDS
