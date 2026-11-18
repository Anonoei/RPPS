"""Every plot draws (Agg backend, tests/unit/conftest.py), and the ones that animate update the line they returned"""
import numpy as np
import pytest

pytest.importorskip("matplotlib")
import matplotlib.pyplot as plt

import rpps as rp
from rpps.viz.plots import freq, time, eye


@pytest.fixture
def x():
    rng = np.random.default_rng(0)
    syms = np.exp(1j*np.pi/4*(2*rng.integers(0, 4, 256) + 1))
    return np.repeat(syms, 4).astype(np.complex64) + 0.05*(rng.standard_normal(1024) + 1j*rng.standard_normal(1024))

@pytest.fixture(autouse=True)
def close():
    yield
    plt.close("all")

@pytest.mark.parametrize("fn", [time.I, time.Q, time.mag, time.pha, freq.mag, freq.pha, freq.psdreal])
def test_line_plots_draw_and_update(fn, x):
    ln = fn(None, x)
    assert len(ln.get_xdata()) == len(ln.get_ydata()) > 0
    assert fn(ln, x[::-1]) is ln

def test_psd_and_signals(x):
    ln = freq.psd(None, x, Fs=1e6, cf=100e6)
    assert freq.psd(ln, x, Fs=1e6, cf=100e6) is ln
    sv = rp.analysis.survey(x, 1e6)
    freq.psd_signals(None, x, list(sv), Fs=1e6)
    freq.mag_phase(None, None, x)
    freq.psd_minmax(plt.subplots()[1], x, 1e6, cache={})

def test_iq_eye_phasor_spectrogram_survey(x):
    lines = time.IQ(None, x)
    time.IQ(lines, x)
    time.IQ3d(None, x[:200])
    eye.IQ(None, x, 4)
    rp.viz.phasor(None, x[::4])
    rp.viz.spectrogram.spectrogram(None, x.reshape(8, -1), Fs=1e6)
    rp.viz.survey(None, rp.analysis.survey(x, 1e6))
