"""Channel models, and the synchronizers against them"""
import numpy as np
import pytest

import rpps as rp

def shaped_qpsk(rng, n=8000, sps=4, beta=0.35):
    sym = (rng.choice([-1, 1], n) + 1j*rng.choice([-1, 1], n))/np.sqrt(2)
    up = np.zeros(n*sps, complex)
    up[::sps] = sym
    h = rp.signal.filters.ShapingRRC(None, sps, beta).pulse
    x = np.convolve(np.convolve(up, h, mode="same"), h, mode="same")
    return x.astype(np.complex64), sym

def qpsk_evm(o):
    """EVM [dB] of QPSK-at-45-degrees symbols after removing a constant rotation"""
    o = o/np.sqrt(np.mean(np.abs(o)**2))
    o = o*np.exp(-1j*(np.angle(np.mean(o**4)) - np.pi)/4)
    pts = np.array([1, -1, 1j, -1j])*np.exp(1j*np.pi/4)
    return 10*np.log10(np.mean(np.min(np.abs(o[:, None] - pts[None, :]), axis=1)**2))

# --- channel ---
@pytest.mark.parametrize("db", [0, 6, 15])
def test_awgn_sets_the_snr(db, rng):
    x = np.exp(2j*np.pi*0.01*np.arange(100000)).astype(np.complex64)
    y = (rp.Data(x, Fs=1e5) + rp.signal.channel.AWGN(db, seed=3)).data
    noise = y - x
    assert 10*np.log10(1/np.mean(np.abs(noise)**2)) == pytest.approx(db, abs=0.15)

def test_awgn_esn0_uses_the_sps():
    x = np.ones(200000, dtype=np.complex64)
    y = (rp.Data(x, sps=4) + rp.signal.channel.AWGN(esn0_db=10, power=1.0, seed=1)).data
    assert 10*np.log10(1/np.mean(np.abs(y - x)**2)) == pytest.approx(10 - 10*np.log10(4), abs=0.15)

def test_awgn_is_repeatable_and_seeds_differ():
    x = np.ones(1000, dtype=np.complex64)
    a, b, c = ((rp.Data(x) + rp.signal.channel.AWGN(5, seed=s)).data for s in (1, 1, 2))
    assert np.array_equal(a, b) and not np.array_equal(a, c)

def test_cfo_leaves_the_centre_frequency():
    d = rp.Data(np.ones(100, dtype=np.complex64), Fs=1e4, cf=1e6) + rp.signal.channel.CFO(300)
    assert d.cf == 1e6
    f = np.angle(d.data[1:]*np.conj(d.data[:-1]))*1e4/(2*np.pi)
    assert np.allclose(f, 300, atol=1e-2)

def test_fading_has_unit_mean_power_and_the_right_spread():
    x = np.ones(200000, dtype=np.complex64)
    g = (rp.Data(x, Fs=1e4) + rp.signal.channel.Fading(100, seed=5)).data
    assert np.mean(np.abs(g)**2) == pytest.approx(1.0, rel=0.25)
    spec = np.abs(np.fft.fftshift(np.fft.fft(g - g.mean())))**2
    f = np.fft.fftshift(np.fft.fftfreq(len(g), 1e-4))
    inside = spec[np.abs(f) <= 110].sum()/spec.sum()
    assert inside > 0.97 # The spectrum is confined to +-fd

def test_rician_fading_is_less_deep_than_rayleigh():
    x = np.ones(50000, dtype=np.complex64)
    ray = np.abs((rp.Data(x, Fs=1e4) + rp.signal.channel.Fading(50, seed=1)).data)
    ric = np.abs((rp.Data(x, Fs=1e4) + rp.signal.channel.Fading(50, k_db=10, seed=1)).data)
    assert np.std(ric) < np.std(ray)

def test_clock_drift_walks_the_timing():
    ramp = np.arange(100000, dtype=np.float32)
    y = (rp.Data(ramp) + rp.signal.channel.ClockDrift(1000)).data # Reads 1000 ppm faster than the clock
    assert y[-1] == pytest.approx(ramp[len(y)-1]*1.001, rel=1e-4)

# --- synchronizers ---
@pytest.mark.parametrize("ted", ["mm", "gardner"])
@pytest.mark.parametrize("delay", [0.0, 0.37, 0.8])
def test_timing_loops_lock(ted, delay, rng):
    x, _ = shaped_qpsk(rng)
    t = np.arange(len(x))
    x = (np.interp(t - delay, t, x.real) + 1j*np.interp(t - delay, t, x.imag)).astype(np.complex64)
    x = (rp.Data(x, Fs=4.0, sps=4) + rp.signal.channel.AWGN(25, power=None, seed=1))
    out = (x + rp.sync.loops.MM(0.05, ted=ted)).data
    assert qpsk_evm(out[1000:]) < -20

@pytest.mark.parametrize("ted", ["mm", "gardner"])
def test_timing_loops_follow_clock_drift(ted, rng):
    x, _ = shaped_qpsk(rng, n=20000)
    d = rp.Data(x, Fs=4.0, sps=4) + rp.signal.channel.TimingOffset(0.5) + rp.signal.channel.ClockDrift(300) + rp.signal.channel.AWGN(25, power=None, seed=1)
    out = (d + rp.sync.loops.MM(0.05, ted=ted)).data
    for a, b in ((2000, 6000), (8000, 13000), (14000, 19000)):
        assert qpsk_evm(out[a:b]) < -18 # Still locked 20000 symbols * 300 ppm = 6 symbols of drift later

@pytest.mark.parametrize("M", [2, 4, 8])
def test_mpower_acquires_psk_of_any_order_with_an_offset(M, rng):
    """Over 20 seeds the worst phase std after acquiring is 0.041 rad"""
    n = 8000
    off = np.pi/4 if M == 4 else 0.0
    pts = np.exp(1j*(off + 2*np.pi*np.arange(M)/M))
    s = pts[rng.integers(0, M, n)].astype(np.complex64)
    y = rp.Data(s, sps=1, Fs=1e3) + rp.signal.channel.CFO(2.0) + rp.signal.channel.PhaseOffset(0.5) + rp.signal.channel.AWGN(25, seed=1)
    out = np.asarray((y + rp.sync.phase.MPower(order=M, offset=float(np.angle(np.mean(pts**M))), window=0.0, min_symbols=256)).data)[n//2:]
    ang = np.angle(out) - off
    step = 2*np.pi/M
    assert np.std(ang - np.round(ang/step)*step) < 0.08

def test_ddcarrier_follows_apsk_through_drift_and_phase_noise(rng):
    """16APSK, a phase that starts 0.1 rad off, drifts 2e-4 cycles a symbol and wanders (0.005 rad a symbol): no rotation
    slip and an rms error near 0.08 rad (over 40 seeds: none slipped, the median 0.082, the worst under 0.11)"""
    from rpps.sync.phase import decided
    pts = np.asarray(rp.symbol.load("apsk/dvbs2_16apsk", rate="2/3").points)
    n = 8000
    s = pts[rng.integers(0, len(pts), n)]
    ph = 0.1 + 2*np.pi*2e-4*np.arange(n) + np.cumsum(rng.normal(0, 0.005, n))
    y = s*np.exp(1j*ph) + 0.05*(rng.normal(size=n) + 1j*rng.normal(size=n))
    out = decided.follow(y, pts)
    assert np.mean(decided.decide(out[200:], pts) != s[200:]) < 0.005
    assert np.sqrt(np.mean(np.angle(out[200:]*np.conj(s[200:]))**2)) < 0.12

def test_carrier_costas_builds_the_loop():
    rx = rp.profile.Profile("x", {"steps": [{"sync": "carrier", "method": "costas", "order": 4}]}).rx()
    assert isinstance(rx.ops[0][1], rp.sync.loops.Costas) and rx.ops[0][1].order == 4

@pytest.mark.parametrize("M", [2, 4, 8, 16])
def test_costas_locks_psk_of_any_order(M, rng):
    n = 8000
    off = np.pi/4 if M == 4 else 0.0
    pts = np.exp(1j*(off + 2*np.pi*np.arange(M)/M))
    s = pts[rng.integers(0, M, n)].astype(np.complex64)
    y = rp.Data(s, sps=1) + rp.signal.channel.CFO(0.002) + rp.signal.channel.PhaseOffset(0.5) + rp.signal.channel.AWGN(25, seed=1)
    out = (y + rp.sync.loops.Costas.from_bandwidth(0.03, M, offset=off)).data[n//2:]
    ang = np.angle(out) - off
    step = 2*np.pi/M
    assert np.std(ang - np.round(ang/step)*step) < 0.15

def _dd_loop(y, points, k1, k2):
    """The second-order decision-directed loop Costas(points=) runs (analysis.burst's own loop, before it used Costas, then DDCarrier)"""
    out, ph, nu = np.empty_like(y), 0.0, 0.0
    for i, v in enumerate(y):
        r = v*np.exp(-1j*ph)
        e = np.angle(r*np.conj(points[np.argmin(np.abs(points - r))]))
        out[i] = r
        nu += k2*e
        ph += k1*e + nu
    return out

def test_costas_on_points_locks_apsk_and_is_the_decision_directed_loop(rng):
    n = 6000
    pts = np.concatenate((np.exp(1j*(np.pi/4 + np.pi/2*np.arange(4))), 2.6*np.exp(1j*(np.pi/12 + np.pi/6*np.arange(12))))) # 16APSK
    pts /= np.sqrt(np.mean(np.abs(pts)**2))
    s = pts[rng.integers(0, len(pts), n)]
    y = (rp.Data(s.astype(np.complex64), sps=1) + rp.signal.channel.CFO(0.001) + rp.signal.channel.PhaseOffset(0.05)
         + rp.signal.channel.AWGN(25, seed=1)).data.astype(np.complex128)
    loop = rp.sync.loops.Costas(0.05, 0.002, points=pts)
    out = loop.step(y, loop.new_state(None), None)
    assert np.allclose(out, _dd_loop(y, pts, 0.05, 0.002), atol=1e-9)
    tail = out[n//2:]
    near = pts[np.argmin(np.abs(tail[:, None] - pts[None, :]), axis=1)]
    assert np.std(np.angle(tail*np.conj(near))) < 0.1


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
    from test_stream import cuts, run
    env = np.where((np.arange(20000)//3000) % 2 == 0, 0.02, 1.5)
    x = (env*(rng.standard_normal(20000) + 1j*rng.standard_normal(20000))).astype(np.complex64)
    want = _peak_ref(x.astype(np.complex128), 1.0, decay)
    for cut in cuts(len(x)):
        got = run(lambda: rp.sync.PeakAGC(1.0, decay), x, cut, Fs=2e6)
        assert np.allclose(got, want, rtol=1e-5, atol=1e-6), cut[:2]
        got = run(lambda: rp.sync.loops.PeakAGC(1.0, decay), x, cut, Fs=2e6) # the loop it replaced
        assert np.allclose(got, want, rtol=1e-5, atol=1e-6), cut[:2]

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

def test_channelizer_separates_tones_and_is_block_independent(rng):
    M, Fs = 8, 8000.0
    n = np.arange(64000)
    tones = {1: 1000.0 + 50, 3: 3000.0 - 80, 6: -2000.0 + 20} # In channels 1, 3 and 6 (= -2)
    x = sum(np.exp(2j*np.pi*f*n/Fs) for f in tones.values()).astype(np.complex64)
    whole = rp.Data(x, Fs=Fs) + rp.signal.filters.Channelizer(M)
    assert whole.Fs == Fs/M and whole.data.shape[1] == M
    power = np.mean(np.abs(whole.data[100:])**2, axis=0)
    assert set(np.nonzero(power > 0.5)[0]) == {1, 3, 6}
    # Pieces of any length give the same bands
    pipe = rp.signal.filters.Channelizer(M)
    stream, outs = None, []
    for a, b in ((0, 777), (777, 5000), (5000, 5001), (5001, len(x))):
        d = rp.Data(x[a:b], Fs=Fs, stream=stream) + pipe
        stream = d.stream
        outs.append(d.data)
    assert np.allclose(np.concatenate(outs), whole.data, atol=1e-5)
