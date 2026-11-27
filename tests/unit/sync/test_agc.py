"""Gain control: the AGC stages, their loops and the peak AGC"""
import numpy as np
import pytest

import rpps as rp
from support.blocks import cuts, run

def test_agc_settles_on_the_target(rng):
    x = (0.05*np.exp(2j*np.pi*0.05*np.arange(20000))).astype(np.complex64)
    out = (rp.Data(x) + rp.sync.AGC(1.0)).data
    assert np.mean(np.abs(out[-2000:])) == pytest.approx(1.0, rel=0.05)
    out2 = (rp.Data(40*x) + rp.sync.AGC(1.0)).data
    assert np.mean(np.abs(out2[-2000:])) == pytest.approx(1.0, rel=0.05)

def _level(v):
    return float(np.exp(np.mean(np.log(np.abs(v)))))

def test_agc_settles_after_a_level_step_without_a_tail(rng):
    """A 30x step: the window holding it is off, the rest on target (over 20 seeds |log level| per 500 samples is under
    0.0096 on both sides; the loop this replaced was 4% off for a thousand samples after the step)"""
    n = 20000
    x = np.concatenate([0.1*(rng.standard_normal(n) + 1j*rng.standard_normal(n)),
                        3*(rng.standard_normal(n) + 1j*rng.standard_normal(n))]).astype(np.complex64)
    y = np.asarray((rp.Data(x, Fs=1e6, flush=True) + rp.sync.AGC(1.0)).data)
    assert len(y) == len(x)
    for k in list(range(0, n - 500, 500)) + list(range(n + 200, 2*n, 500)):
        assert abs(np.log(_level(y[k:k + 500]))) < 0.03, k

def test_agc_takes_no_loop_rate():
    with pytest.raises(TypeError):
        rp.sync.AGC(1.0, 0.01) # (a loop's mu in second place is refused, not taken for a window)

def test_agc_loop_settles_on_the_target(rng):
    n = 20000
    x = np.concatenate([0.1*(rng.standard_normal(n) + 1j*rng.standard_normal(n)),
                        3*(rng.standard_normal(n) + 1j*rng.standard_normal(n))]).astype(np.complex64)
    y = np.asarray((rp.Data(x, Fs=1e6) + rp.sync.loops.AGC(1.0, 0.01)).data)
    assert abs(np.log(_level(y[n - 2000:n]))) < 0.1 and abs(np.log(_level(y[-2000:]))) < 0.1

def test_profile_agc_names_the_loop():
    ctx = rp.profile.steps.context()
    build = lambda m: rp.profile.steps.build({"sync": "agc", **m}, rp.Rep.SAMPLES, ctx)
    with pytest.raises(rp.errors.ConfigError, match="loop"):
        build({"mu": 0.01})
    assert isinstance(build({"method": "loop", "mu": 0.01}).obj, rp.sync.loops.AGC)

def _peak_ref(x, target, decay, floor=1e-6):
    level, out = 0.0, np.empty(len(x))
    for i, m in enumerate(np.abs(x).astype(np.float64)):
        level = m if m > level*decay else level*decay
        out[i] = level
    return x*(target/np.maximum(out, floor))

@pytest.mark.parametrize("decay", [0.999, 0.9995, 1.0])
def test_peak_agc_is_the_running_peak_for_any_cut(decay, rng):
    env = np.where((np.arange(20000)//3000) % 2 == 0, 0.02, 1.5)
    x = (env*(rng.standard_normal(20000) + 1j*rng.standard_normal(20000))).astype(np.complex64)
    want = _peak_ref(x.astype(np.complex128), 1.0, decay)
    for cut in cuts(len(x)):
        got = run(lambda: rp.sync.PeakAGC(1.0, decay), x, cut, Fs=2e6)
        assert np.allclose(got, want, rtol=1e-5, atol=1e-6), cut[:2]
        got = run(lambda: rp.sync.loops.PeakAGC(1.0, decay), x, cut, Fs=2e6) # the loop it replaced
        assert np.allclose(got, want, rtol=1e-5, atol=1e-6), cut[:2]

def test_peak_agc_scales_each_burst_by_its_own_peak():
    x = np.concatenate((np.zeros(10), 0.3*np.array([1, 0, 1, 1, 0]*4), np.zeros(200), 2.0*np.array([1, 1, 0, 1, 0]*4))).astype(np.float32)
    out = np.asarray((rp.Data(x, Fs=1.0) + rp.sync.PeakAGC(target=1.0, decay=0.999)).data)
    assert np.allclose(out[10:30][x[10:30] > 0], 1.0, atol=1e-5) and np.allclose(out[-20:][x[-20:] > 0], 1.0, atol=0.03)
    assert np.allclose(out[x == 0], 0.0)

def test_agc_lin_is_linear_agc():
    a = rp.sync.AGC(2.0, log=False)
    x = (0.1*np.exp(1j*np.linspace(0, 50, 20000))).astype(np.complex64)
    y = rp.Data(x, Fs=1e6) + a
    assert np.mean(np.abs(y.data[-2000:])) == pytest.approx(2.0, rel=0.05)
