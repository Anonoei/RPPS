"""Effort levels (rpps._effort) and the timing interpolation they choose (rpps.math.interp.fractional)"""
import numpy as np
import pytest

import rpps as rp
from rpps import _effort
from rpps.math import interp
from support.signals import bpsk_llr

P = np.exp(1j*np.pi/4*(2*np.arange(4) + 1))

def test_levels_set_and_nest():
    assert _effort.level() == "balanced"
    with rp.effort("max"):
        assert _effort.level() == "max" and _effort.scale("iters", 50) == 200 and _effort.choice("maxlog", True) is False
        with rp.effort("fast"):
            assert _effort.scale("iters", 50) == 25 and _effort.choice("maxlog", False) is True and _effort.interp_taps(8) == 4
        assert _effort.level() == "max"
    assert _effort.level() == "balanced" and _effort.scale("iters", 50) == 50 and _effort.choice("maxlog", True) is True
    assert _effort.interp_taps(2) == 8 and _effort.interp_taps(4) == 4 # balanced: cubic from 4 samples a symbol
    assert _effort.interp_taps(1.67) == 12 # (a signal sampled close to its width: the long sinc)
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

@pytest.mark.parametrize("u", [2, 4, 8])
def test_upsampled_and_edges(u):
    """x at 1/u samples, band limited: a tone at 0.13 cycles a sample within 1e-3 at every phase"""
    x = np.exp(2j*np.pi*0.13*np.arange(500))
    y = interp.upsampled(x, 100, 300, 0, u)
    assert len(y) == u*(300 - interp.REACH - 100)
    assert np.max(np.abs(y - np.exp(2j*np.pi*0.13*(100 + np.arange(len(y))/u)))) < 1e-3
    with pytest.raises(ValueError):
        interp.upsampled(x, 100, 300, 0, 3) # (not a row of the sinc table)
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
    c = rp.symbol.load("psk/qpsk", "dvb-s2")
    ld = rp.code.load("block/ldpc/dvbs2_64800_1_2").decoder
    rng = np.random.default_rng(5)
    s = np.asarray((rp.Data(np.zeros(ld.den, bool))*c).data)
    llr = c.llr(s + np.sqrt(10**(-0.9/10)/2)*(rng.standard_normal(len(s)) + 1j*rng.standard_normal(len(s))))
    with rp.effort("max"):
        assert not np.any(ld.decode(llr)) # 0.9 dB: 200 iterations get there (the definition's 50 mostly don't)
    tb = rp.code.load("block/turbo/lte", K=40).decoder
    m = rng.integers(0, 2, 40).astype(bool)
    x = 1.0 - 2.0*rp.code.load("block/turbo/lte", K=40).encoder.encode(m)
    for lvl in rp._effort.LEVELS:
        with rp.effort(lvl):
            assert np.array_equal(tb.decode(4*x)[:40].astype(bool), m)

def frame_errors(code, msgs, llr):
    out = np.asarray(code.decoder.decode(llr)).astype(bool)[:msgs.size].reshape(msgs.shape)
    return int(np.count_nonzero((out != msgs).any(axis=1)))

def test_new_settings_follow_the_level():
    with rp.effort("max"):
        assert _effort.scale("list", 8) == 32 and _effort.scale("list", 1) == 4 and _effort.shift("chase", 4) == 5
        assert _effort.choice("check", "minsum") == "sumproduct"
    with rp.effort("fast"):
        assert _effort.scale("list", 8) == 4 and _effort.scale("list", 1) == 1 and _effort.shift("chase", 4) == 3
        assert _effort.choice("check", "minsum") == "minsum"
    assert _effort.scale("list", 8) == 8 and _effort.shift("chase", 4) == 4 and _effort.choice("check", "minsum") == "minsum"

def test_polar_list_follows_the_level():
    code = rp.code.load("block/polar/256_128_crc16")
    rng = np.random.default_rng(2)
    m = rng.integers(0, 2, (200, code.encoder.num)).astype(bool)
    llr = bpsk_llr(code.encoder.encode(m.reshape(-1)), 2.0, code.encoder.rate, rng)
    errs = {}
    for lvl in _effort.LEVELS:
        with rp.effort(lvl):
            errs[lvl] = frame_errors(code, m, llr)
    assert errs["max"] < errs["balanced"] < errs["fast"], errs # lists of 32, 8, 4

def test_tpc_follows_the_level():
    code = rp.code.load("block/product/tpc_64_57_sq")
    rng = np.random.default_rng(4)
    m = rng.integers(0, 2, (24, code.encoder.num)).astype(bool)
    llr = bpsk_llr(code.encoder.encode(m.reshape(-1)), 3.2, code.encoder.rate, rng)
    errs = {}
    for lvl in _effort.LEVELS:
        with rp.effort(lvl):
            errs[lvl] = frame_errors(code, m, llr)
    assert errs["max"] <= errs["balanced"] < errs["fast"], errs # Chase 5 x 16 rounds, 4 x 4, 3 x 2

def test_ldpc_sum_product_at_max():
    """DVB-S2 16200 1/2 at Eb/N0 0.8 dB: min-sum (even 200 iterations) loses most words, sum-product none"""
    code = rp.code.load("block/ldpc/dvbs2_16200_1_2")
    rng = np.random.default_rng(3)
    m = rng.integers(0, 2, (8, code.encoder.num)).astype(bool)
    llr = bpsk_llr(code.encoder.encode(m.reshape(-1)), 0.8, code.encoder.rate, rng)
    with rp.effort("balanced"):
        ms = frame_errors(code, m, llr)
    with rp.effort("max"):
        assert frame_errors(code, m, llr) == 0
    assert ms >= 2
