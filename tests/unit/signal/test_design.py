"""Filter design: windowed, Kaiser, Remez and RRC designers"""
import numpy as np
import pytest

import rpps as rp

@pytest.mark.oracle
def test_designers_match_scipy():
    scipy_signal = pytest.importorskip("scipy.signal")
    from rpps.signal.filters.impl.designer import designer
    for cut, pz in ((0.2, True), (0.2, False), ([0.2, 0.4], False), ([0.2, 0.4], True)):
        assert np.allclose(designer(51, np.array(cut, dtype=float), pass_zero=pz, fs=2),
                           scipy_signal.firwin(51, cut, pass_zero=pz), atol=1e-6)

def test_band_filters_pass_and_stop_the_right_bands():
    Fs = 10e3
    t = np.arange(20000)/Fs
    tones = {f: np.exp(2j*np.pi*f*t).astype(np.complex64) for f in (500, 1500, 3000)}
    def gain(stage, f):
        y = (rp.Data(tones[f], Fs=Fs) + stage).data[400:]
        return np.mean(np.abs(y))
    bp = rp.signal.filters.band_pass(201, 1000, 2000, Fs)
    bs = rp.signal.filters.band_stop(201, 1000, 2000, Fs)
    assert gain(bp, 1500) > 0.9 and gain(bp, 500) < 0.05 and gain(bp, 3000) < 0.05
    assert gain(bs, 1500) < 0.05 and gain(bs, 500) > 0.9 and gain(bs, 3000) > 0.9

def test_rrc_is_zero_isi_and_exactly_the_taps_asked_for():
    for taps in (5, 21, 101):
        assert len(rp.signal.filters.ShapingRRC(taps, 4).pulse) == len(rp.signal.filters.ShapingRC(taps, 4).pulse) == taps
    h = rp.signal.filters.ShapingRRC(None, 8, 0.25).pulse
    rc = np.convolve(h, h)
    c = len(rc)//2
    assert np.allclose(rc[c-24:c+25:8]/rc[c], [0, 0, 0, 1, 0, 0, 0], atol=2e-3)
    for beta in (0.25, 1.0): # beta = 1 puts samples on the t = 1/(4 beta) singularity
        assert np.isfinite(rp.signal.filters.ShapingRRC(41, 4, beta).pulse).all()
        assert np.isfinite(rp.signal.filters.ShapingRC(41, 4, beta).pulse).all()

def test_kaiser_design_meets_its_specification():
    lp = rp.signal.filters.low_pass_kaiser(2000, 400, atten_db=60, fs=20e3)
    h = lp.filt.astype(float)
    H = np.abs(np.fft.rfft(h, 8192))
    f = np.fft.rfftfreq(8192, 1/20e3)
    assert H[f < 1700].min() > 0.99 and H[f < 1700].max() < 1.01
    assert 20*np.log10(H[f > 2300].max()) < -58

def test_filters_take_a_window():
    for w in (np.blackman, np.hanning, rp.signal.filters.kaiser_window(8.0)):
        f = rp.signal.filters.low_pass(101, 1000, 10e3, window=w).filt
        assert abs(f.sum() - 1) < 1e-3

def test_remez_meets_the_spec_without_scipy():
    h = rp.signal.filters.remez(101, [0, 1000, 1500, 5000], [1, 0], fs=10e3).filt
    H = np.abs(np.fft.rfft(h, 4096)); f = np.fft.rfftfreq(4096, 1e-4)
    assert H[f < 900].min() > 0.95 and H[f > 1600].max() < 0.05

def test_high_pass_kaiser_blocks_dc():
    f = rp.signal.filters.high_pass_kaiser(0.25, 0.05)
    assert abs(np.sum(f.filt)) < 1e-2

def test_shaping_mat_rrc_shapes_a_block():
    f = rp.signal.filters.ShapingMatRRC(16, 4)
    assert f.pulse.shape == (16, 64)
    assert f.burst(np.ones(40)).shape == (160,)

@pytest.mark.parametrize("shape", ["sinc", "exp", "sqr"])
def test_prototype_low_and_high_pass_split_the_band(shape):
    """The fixed-shape prototypes: the low pass passes DC, its high pass (an impulse less the low pass) blocks it"""
    from rpps.signal.filters.impl import designer
    low, high = designer.lowpass(64, shape), designer.highpass(64, shape)
    assert len(low) == len(high) == 64
    assert np.sum(low) == pytest.approx(1.0) and np.sum(high) == pytest.approx(0.0, abs=1e-9)
    assert abs(np.sum(low*(-1)**np.arange(64))) < 0.2 < abs(np.sum(high*(-1)**np.arange(64)))

