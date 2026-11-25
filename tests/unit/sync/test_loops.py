"""Per-sample synchronizers: timing loops, Costas and the decision-directed carrier"""
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
