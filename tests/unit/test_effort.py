"""Effort levels (rpps._effort) and the timing interpolation they choose (rpps.math.interp.fractional)"""
import numpy as np
import pytest

import rpps as rp
from rpps import _accel, _effort
from rpps.math import interp

P = np.exp(1j*np.pi/4*(2*np.arange(4) + 1))


def test_levels_set_and_nest():
    assert _effort.level() == "balanced"
    with rp.effort("max"):
        assert _effort.level() == "max" and _effort.iters(50) == 200 and _effort.maxlog(True) is False
        with rp.effort("fast"):
            assert _effort.iters(50) == 25 and _effort.maxlog(False) is True and _effort.interp_taps(8) == 4
        assert _effort.level() == "max"
    assert _effort.level() == "balanced" and _effort.iters(50) == 50 and _effort.maxlog(True) is True
    assert _effort.interp_taps(2) == 8 and _effort.interp_taps(4) == 4 # balanced: cubic from 4 samples a symbol
    with pytest.raises(ValueError):
        rp.set_effort("extreme")

@pytest.mark.parametrize("taps, floor", [(8, -39), (12, -55)])
def test_fractional_is_band_limited(taps, floor):
    """Tones up to the band edge of a matched filter's output at 2 samples a symbol (0.34 cycles a sample)"""
    t = 50 + np.random.default_rng(0).random(3000)*400
    for f in (0.05, 0.2, 0.34):
        x = np.exp(2j*np.pi*f*np.arange(600))
        err = 20*np.log10(np.sqrt(np.mean(np.abs(interp.fractional(x, t, taps) - np.exp(2j*np.pi*f*t))**2)))
        assert err < floor, (f, err)

def test_fractional_native_matches_numpy():
    if not _accel.available():
        pytest.skip("rpps._native is not built")
    rng = np.random.default_rng(1)
    x = (rng.standard_normal(1000) + 1j*rng.standard_normal(1000)).astype(np.complex64)
    t = 10 + rng.random(500)*900
    nat = interp.fractional(x, t, 12)
    saved, _accel.native = _accel.native, None
    try:
        ref = interp.fractional(x, t, 12)
    finally:
        _accel.native = saved
    np.testing.assert_allclose(nat, ref, atol=1e-5)

def test_doubled_and_edges():
    x = np.exp(2j*np.pi*0.13*np.arange(500))
    y = interp.doubled(x, 100, 300, 0)
    assert len(y) == 2*(300 - interp.REACH - 100)
    assert np.max(np.abs(y - np.exp(2j*np.pi*0.13*(100 + np.arange(len(y))/2)))) < 1e-3
    z = interp.fractional_edge(x[:20], np.array([0.5, 19.5]), 12) # beyond the ends: zeros
    assert np.all(np.isfinite(z))

def qpsk_matched(n, sps, delay, beta=0.35, seed=0):
    """QPSK through RRC and its matched filter, delayed exactly (band limited) by `delay` samples"""
    s = P[np.random.default_rng(seed).integers(0, 4, n)]
    h = rp.signal.filters.ShapingRRC(None, sps, beta).pulse
    up = np.zeros(n*sps, complex)
    up[::sps] = s
    tx = np.convolve(up, h)
    rx = np.fft.ifft(np.fft.fft(tx)*np.exp(-2j*np.pi*np.fft.fftfreq(len(tx))*delay))
    return (np.convolve(rx, h)/np.sum(h*h)).astype(np.complex64)

def evm_db(y):
    y = y/np.sqrt(np.mean(np.abs(y)**2))
    return 10*np.log10(np.mean(np.min(np.abs(y[:, None] - P[None, :])**2, axis=1)))

@pytest.mark.parametrize("level, worst", [("fast", -25), ("balanced", -45), ("max", -52)])
def test_oerder_meyr_two_samples_a_symbol(level, worst):
    """At 2 samples a symbol the square-law line sits at Nyquist and a quarter-symbol delay cancelled it: acquired now,
    and the symbols as clean as the level's interpolator allows"""
    x = qpsk_matched(30000, 2, 0.5)
    with rp.effort(level):
        d = rp.Data(x, Fs=2e3, sps=2, flush=True) + rp.sync.time.OerderMeyr(2)
    assert d.stream.est["timing"]["phase"] == pytest.approx(0.5, abs=0.02)
    assert evm_db(np.asarray(d.data)[8000:-100]) < worst

def test_effort_change_mid_stream_and_any_cut():
    x = qpsk_matched(12000, 2, 0.3)
    def run(cut, levels):
        stage, st, out, i = rp.sync.time.OerderMeyr(2), None, [], 0
        for b, lvl in zip(cut, levels):
            with rp.effort(lvl):
                d = rp.Data(x[i:i+b], stream=st, Fs=2e3, sps=2, flush=i + b >= len(x))
                st = d.stream
                out.append(np.asarray((d + stage).data))
            i += b
        return np.concatenate(out)
    whole = run([len(x)], ["max"])
    np.testing.assert_allclose(run([1, 777, 9000, len(x) - 9778], ["max"]*4), whole, atol=1e-5)
    mixed = run([6000, len(x) - 6000], ["fast", "max"]) # no symbol lost or repeated where the level changes
    assert len(mixed) == len(whole)

def test_ldpc_and_turbo_follow_the_level():
    c = rp.symbol.load("dvbs2_qpsk")
    ld = rp.code.load("block", "ldpc/dvbs2_64800_1_2").decoder
    rng = np.random.default_rng(5)
    s = np.asarray((rp.Data(np.zeros(ld.den, bool))*c).data)
    llr = c.llr(s + np.sqrt(10**(-0.9/10)/2)*(rng.standard_normal(len(s)) + 1j*rng.standard_normal(len(s))))
    with rp.effort("max"):
        assert not np.any(ld.decode(llr)) # 0.9 dB: 200 iterations get there (the definition's 50 mostly don't)
    tb = rp.code.load("block", "turbo/lte", K=40).decoder
    m = rng.integers(0, 2, 40).astype(bool)
    x = 1.0 - 2.0*rp.code.load("block", "turbo/lte", K=40).encoder.encode(m)
    for lvl in rp._effort.LEVELS:
        with rp.effort(lvl):
            assert np.array_equal(tb.decode(4*x)[:40].astype(bool), m)
