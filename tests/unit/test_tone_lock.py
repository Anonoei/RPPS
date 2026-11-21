"""rp.sync.ToneLock: an oscillator locked to a pilot, the stream mixed with it"""
import numpy as np
import pytest

import rpps as rp

FS = 480e3

def run(x, lock, block, Fs=FS):
    out = list(rp.source.Memory(x, Fs=Fs)(block) + lock)
    return np.concatenate([np.asarray(b.data) for b in out]), out[-1].state(lock)

def test_the_block_cut_does_not_matter(rng):
    n = 200_000
    x = (0.3*np.exp(2j*np.pi*(19e3 + 7.0)/FS*np.arange(n) + 0.4j) + 0.05*rng.standard_normal(n)).astype(np.complex64)
    a, sa = run(x, rp.sync.ToneLock(19e3, win=4096), 4096*5)
    b, sb = run(x, rp.sync.ToneLock(19e3, win=4096), 1001)
    assert np.allclose(a, b, atol=1e-5) and list(sa.errs) == pytest.approx(list(sb.errs), abs=1e-9)

def test_a_complex_tone_off_its_nominal_frequency_ends_at_dc(rng):
    n = 400_000
    df = 25.0 # Hz the tone is off (a radio's clock error)
    x = (np.exp(2j*np.pi*(-50e3 + df)/FS*np.arange(n) + 1.0j) + 0.1*(rng.standard_normal(n) + 1j*rng.standard_normal(n))).astype(np.complex64)
    y, st = run(x, rp.sync.ToneLock(-50e3, win=4096), 65536)
    tail = y[-50_000:]
    assert abs(np.angle(np.mean(tail))) < 0.1 and abs(np.mean(tail)) > 0.9 # At DC, phase locked
    assert st.freq*FS/(2*np.pi) == pytest.approx(-50e3 + df, abs=1.0)
    assert st.level == pytest.approx(1.0, rel=0.05) and abs(st.err) < 0.1

def test_real_input_is_a_product_detector_at_the_harmonic(rng):
    """FM stereo: the 19 kHz pilot locks the oscillator; 2 x cos at 38 kHz brings the L-R subcarrier to baseband"""
    n = 480_000
    t = np.arange(n)/FS
    diff = np.sin(2*np.pi*400*t) # L-R audio
    mpx = (0.1*np.cos(2*np.pi*19e3*t + 0.3) + diff*np.cos(2*np.pi*38e3*t + 0.6)).astype(np.float32)
    y, st = run(mpx, rp.sync.ToneLock(19e3, win=26214, kp=0.3, ki=0.04, harmonic=2, min_level=0.005, lock_level=0.015), 26214)
    assert y.dtype == np.float32 and st.locked and st.level == pytest.approx(0.1, rel=0.05)
    k = np.ones(48)/48 # (a crude low-pass: 38 and 76 kHz products averaged out)
    base = np.convolve(y, k, mode="same")[-100_000:-1000]
    ref = diff[-100_000:-1000]
    assert np.corrcoef(base, ref)[0, 1] > 0.99

def test_no_tone_holds_the_oscillator(rng):
    x = (0.001*rng.standard_normal(100_000)).astype(np.float32)
    _, st = run(x, rp.sync.ToneLock(19e3, win=4096, harmonic=2, min_level=0.005, lock_level=0.015), 8192)
    assert st.freq == pytest.approx(2*np.pi*19e3/FS) and not st.locked
    assert len(st.errs) == 100_000//4096 + 1 # (the last 1696 samples: over min_part of a window, measured at the end)
