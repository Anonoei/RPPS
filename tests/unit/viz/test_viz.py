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
    check = rp.analysis.burst.check_plan(plan, recs)
    assert check.counts["none"] == 1
    rng = np.random.default_rng(0)
    x = (rng.standard_normal(60000) + 1j*rng.standard_normal(60000)).astype(np.complex64)
    ax = rp.viz.bursts(None, check, x=x, Fs=8e6)
    assert ax.collections and ax.patches
    rp.viz.bursts(None, check, fold=True)

def _demodulated(rng, n=40):
    """A demodulate result made by hand: two kinds, each with its known run, two transmitters' offsets and powers"""
    from rpps.analysis.burst import BurstResult, Bursts
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
    B = rp.analysis.burst
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

@pytest.mark.parametrize("sps", [7, 8])
def test_eye_traces_are_centred_on_the_instants(sps):
    """get_eye cuts a fixed grid; get_eye_at puts the sampling instant (fractional too) in the middle column and derotates"""
    x = np.exp(1j*np.pi/4)*np.ones(40*sps + 3, np.complex64)
    grid = eye.get_eye(x, sps)
    assert grid.shape[1] == 2*(sps//2) + 1 and len(grid) == 41
    mesh = eye.get_eye_at(x, sps, 3.5, derotate=4)
    assert mesh.shape[1] % 2 == 1 and len(mesh) > 30
    np.testing.assert_allclose(mesh[:, mesh.shape[1]//2].imag, 0, atol=1e-6)
    assert eye.get_eye_at(x[:sps//2], sps, 0.0).shape[0] == 0

def test_lock_phase_takes_out_a_carrier():
    rng = np.random.default_rng(0)
    k = np.arange(200)
    z = np.exp(1j*(np.pi/2*rng.integers(0, 4, len(k)) + 0.3 + 0.01*k))
    locked = eye.lock_phase(z[:, None], 4)[:, 0]
    assert np.abs(np.angle(locked**4)).max() < 0.2 < np.abs(np.angle(z**4)).max()

@pytest.mark.parametrize("sync", ["oerder_meyr", "mat"])
def test_the_eye_follows_a_timing_sync(sync):
    """sync_phase says where the sync samples in the block it just took: the eye drawn there is open at its centre"""
    rng = np.random.default_rng(0)
    syms = np.exp(1j*np.pi/4*(2*rng.integers(0, 4, 4000) + 1)).astype(np.complex64)
    x = (rp.Data(syms, Fs=1e3, sps=1) + rp.wave.Shape(8, rp.signal.filters.load("pulse/rrc/0p35"))).data
    st = rp.sync.time.OerderMeyr() if sync == "oerder_meyr" else rp.sync.time.Mat()
    stages = st if sync == "oerder_meyr" else rp.sync.Acquire() + st
    stream = None
    for i in range(0, len(x), 4096):
        d = rp.Data(x[i:i + 4096], Fs=8e3, sps=8, stream=stream)
        stream = d.stream
        out = d + stages
    phase = eye.sync_phase(out.state(st), len(d), d.meta.pos, d.sps)
    assert phase is not None
    block = x[-len(d):]
    mesh = eye.get_eye_mf(block, 8, phase, 0.35, lock=4)
    centre = np.abs(mesh[:, mesh.shape[1]//2])
    assert len(mesh) > 100 and np.std(centre) < 0.1*np.mean(centre)
    eye.IQ(None, block, 8, phase=phase, beta=0.35, derotate=4, lock=4)
    eye.IQ(None, block, 8, phase=phase)

@pytest.mark.parametrize("rail", [eye.I, eye.Q])
def test_single_rail_eyes_draw_and_update(rail, x):
    lines = rail(None, x[:400], 4)
    assert len(lines) > 50
    assert rail(lines, x[:400], 4) is lines

def test_sync_phase_is_none_before_the_sync_runs():
    from types import SimpleNamespace
    assert eye.sync_phase(SimpleNamespace(), 100) is None
    assert eye.sync_phase(rp.sync.time.Mat().new_state(rp.base.Meta()), 100) is None

