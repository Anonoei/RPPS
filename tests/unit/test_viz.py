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


def test_eye_is_upright_and_takes_a_block_with_no_traces():
    """Amplitude up (not inverted); a short last block, no whole trace in it, draws nothing instead of failing"""
    eye.IQ(None, np.zeros(0, np.complex64), 8, phase=0.0, beta=0.35)
    lines = eye.IQ(None, np.exp(2j*np.pi*0.01*np.arange(400)).astype(np.complex64), 8)
    lo, hi = lines[0][0].axes.get_ylim() if isinstance(lines, tuple) else lines[0].axes.get_ylim()
    assert lo < hi

def test_bursts_on_their_plan():
    plan = rp.plan.load("demo/mftdma")
    recs = [{"freq": w.freq, "start": w.burst/plan.clock, "length": 0.8*w.length/plan.clock, "kind": None}
            for w in plan.windows(0, 4*plan.sequences[0].duration)[::5]]
    recs.append({"freq": 0.0, "start": 1e-3, "length": 1e-3, "kind": None}) # (no carrier there: in no slot)
    check = rp.analysis.spectrum.burst.check_plan(plan, recs)
    assert check.counts["none"] == 1
    rng = np.random.default_rng(0)
    x = (rng.standard_normal(60000) + 1j*rng.standard_normal(60000)).astype(np.complex64)
    ax = rp.viz.bursts(None, check, x=x, Fs=8e6)
    assert ax.collections and ax.patches
    rp.viz.bursts(None, check, fold=True)

def _demodulated(rng, n=40):
    """A demodulate result made by hand: two kinds, each with its known run, two transmitters' offsets and powers"""
    from rpps.analysis.spectrum.burst import BurstResult, Bursts
    word = np.exp(1j*np.pi/4*(2*rng.integers(0, 4, 32) + 1))
    out = []
    for i in range(n):
        data = np.exp(1j*np.pi/4*(2*rng.integers(0, 4, 200) + 1))
        y = np.concatenate((word, data)) + 0.05*(rng.standard_normal(232) + 1j*rng.standard_normal(232))
        tx = i % 2
        out.append(BurstResult(start=1000.0*i, symbols=y, cfo=1e3*(1 + tx) + rng.normal(0, 30), clock=0.0,
                               gain=10**((60 + 2*tx)/20), evm=0.05, kind=i % 3 == 0, slot=i % 5, frame=0,
                               known=(0, 32), ok=True))
    return Bursts(bursts=out, mod="psk/qpsk", baud=1e5)

def test_burst_timing_stack_terminals_and_plan_evidence():
    rng = np.random.default_rng(0)
    plan = rp.plan.load("demo/mftdma")
    B = rp.analysis.spectrum.burst
    recs = [{"freq": w.freq, "start": w.burst/plan.clock + rng.normal(0, 5e-6), "length": 0.8*w.length/plan.clock,
             "kind": plan.tx_types[w.tx_type].fields["word"]} for w in plan.windows(0, 20*plan.sequences[0].duration)
            if rng.random() < 0.4]
    check = B.check_plan(plan, recs)
    for by in ("carrier", "slot", "kind"):
        assert rp.viz.burst_timing(None, check, by=by).lines
    r = _demodulated(rng)
    for what in ("phase", "magnitude"):
        ax = rp.viz.burst_stack(None, r, kind=False, what=what)
        assert ax.images
    ax = rp.viz.burst_terminals(None, r, x="cfo", y="gain", by="slot")
    assert len(ax.lines) == 5
    rp.viz.burst_terminals(None, r, x="clock", y="evm", by=None)
    g = B.learn_plan(recs)
    a, b = rp.viz.plan_evidence(None, g)
    assert a.lines and b.lines
    mixed = [dict(r_, kind=str(rng.integers(3))) for r_ in recs] # (kinds at random: no frame period is pure)
    with pytest.raises(B.NoPlan) as e:
        B.learn_plan(mixed)
    rp.viz.plan_evidence(None, e.value)
