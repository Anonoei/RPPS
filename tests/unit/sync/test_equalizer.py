"""Equalizers: CMA, LMS with training, fractionally spaced"""
import numpy as np

import rpps as rp

def qpsk_symbols(rng, n):
    return (np.exp(1j*(np.pi/4 + np.pi/2*rng.integers(0, 4, n)))).astype(np.complex64)

# --- equalizers ---
def test_cma_opens_the_eye_through_multipath(rng):
    s = qpsk_symbols(rng, 30000)
    chan = rp.signal.channel.Multipath([1.0, 0.45, 0.25j], [0, 1, 2])
    x = rp.Data(s, sps=1) + chan + rp.signal.channel.AWGN(30, seed=1)
    ref = np.abs(x.data[100:2000])
    out = (x + rp.sync.loops.CMA(9, 0.002)).data
    # Modulus spread: before, the ISI makes |x| scatter; after, it sits near 1
    assert np.std(np.abs(out[-3000:])) < 0.5*np.std(ref)

def test_lms_with_training_converges(rng):
    s = qpsk_symbols(rng, 6000)
    x = rp.Data(s, sps=1) + rp.signal.channel.Multipath([1.0, 0.4], [0, 1])
    eq = rp.sync.loops.LMS(7, 0.02, points=[1/np.sqrt(2)*(a + 1j*b) for a in (1, -1) for b in (1, -1)])
    out = (x + eq).data
    pts = np.array([a + 1j*b for a in (1, -1) for b in (1, -1)])/np.sqrt(2)
    tail = out[-1000:]
    err = np.min(np.abs(tail[:, None] - pts[None, :]), axis=1)
    assert np.mean(err) < 0.2

def test_fractionally_spaced_equalizer_halves_the_rate(rng):
    x = rp.Data((rng.normal(size=2000) + 1j*rng.normal(size=2000)).astype(np.complex64), Fs=2e4, sps=2)
    out = x + rp.sync.loops.CMA(9, 0.001)
    assert len(out) == 1000 and out.Fs == 1e4 and out.sps == 1

def test_lms_takes_the_detected_constellation(rng):
    s = qpsk_symbols(rng, 6000)
    x = rp.Data(s, sps=1) + rp.signal.channel.Multipath([1.0, 0.4], [0, 1])
    eq = rp.sync.loops.LMS(7, 0.02) # No points: they come from the detection
    assert len(x + eq) == 0 # Nothing detected yet: nothing to decide against, nothing out
    x.stream.det = type("Det", (), {"mod": "qpsk"})()
    out = (x + eq).data
    pts = np.array([a + 1j*b for a in (1, -1) for b in (1, -1)])/np.sqrt(2)
    assert np.mean(np.min(np.abs(out[-1000:, None] - pts[None, :]), axis=1)) < 0.2
