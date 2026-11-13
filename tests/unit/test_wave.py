"""Waveform layer: shaping pulses, Stagger (O-QPSK), FM (FSK/GFSK) and Decimate's offset"""
import numpy as np
import pytest

import rpps as rp
from rpps.signal.filters import shaping

def stream(x, blocks, stage, **rates):
    """x cut into `blocks` through one stage in one stream"""
    outs, s, i = [], None, 0
    for b in blocks:
        d = rp.Data(x[i:i + b], stream=s, **rates)
        s = d.stream
        outs.append((d + stage).data)
        i += b
    return np.concatenate(outs)

CUTS = lambda n: ([n], [n//3, n//3, n - 2*(n//3)], [1]*20 + [n - 20], [7, 0, n - 7])

# ---- pulses

def test_halfsine_pulse():
    for sps in (2, 4, 8):
        p = shaping.ShapingHalfSine(sps).pulse
        assert len(p) == sps + 1 and p[0] == pytest.approx(0, abs=1e-12) and p[-1] == pytest.approx(0, abs=1e-12)
        assert p.max() == pytest.approx(1.0) and np.argmax(p) == sps//2
        assert np.allclose(p, p[::-1])
    with pytest.raises(ValueError):
        shaping.ShapingHalfSine(1)

def test_halfsine_config_uses_stream_sps():
    st = rp.signal.filters.load("pulse", "halfsine/halfsine")
    x = np.zeros(32, np.complex64)
    x[::8] = 1
    out = (rp.Data(x, Fs=8e6, sps=8) + st).data
    assert np.allclose(out[:9].real, shaping.ShapingHalfSine(8).pulse, atol=1e-6)

@pytest.mark.parametrize("name,bt", [("gaussian/bt0p5", 0.5), ("gaussian/bt0p3", 0.3)])
def test_gaussian_pulse_holds_a_level(name, bt):
    """The taps sum to sps (a held level comes out as itself) and a narrower BT spreads the pulse over more symbols"""
    sps = 8
    p = shaping.ShapingGaussian(sps, bt).pulse
    assert p.sum() == pytest.approx(sps)
    held = np.convolve(np.ones(40), p/sps)
    assert held[len(held)//2] == pytest.approx(1.0, abs=1e-3)
    wide = shaping.ShapingGaussian(sps, 0.3).pulse
    narrow = shaping.ShapingGaussian(sps, 1.0).pulse
    assert wide.max() < narrow.max()
    assert rp.signal.filters.load("pulse", name) is not None

# ---- Decimate offset

def test_decimate_offset_across_blocks():
    x = np.arange(100, dtype=np.float32) + 0j
    for cuts in CUTS(100):
        out = stream(x.astype(np.complex64), cuts, rp.signal.sample.down.Decimate(8, offset=3), Fs=8.0, sps=8)
        assert np.array_equal(out.real, x.real[3::8])

# ---- Stagger

def test_stagger_delays_q_by_half_a_symbol():
    rng = np.random.default_rng(0)
    x = (rng.standard_normal(80) + 1j*rng.standard_normal(80)).astype(np.complex64)
    out = stream(x, [80], rp.wave.Stagger(), Fs=8.0, sps=8)
    assert np.array_equal(out.real, x.real)
    assert np.array_equal(out.imag[4:], x.imag[:-4]) and np.all(out.imag[:4] == 0)

def test_stagger_rx_realigns():
    rng = np.random.default_rng(1)
    x = (rng.standard_normal(200) + 1j*rng.standard_normal(200)).astype(np.complex64)
    tx = stream(x, [200], rp.wave.Stagger(), Fs=8.0, sps=8)
    rx = stream(tx, [200], rp.wave.Stagger(rail="i"), Fs=8.0, sps=8)
    assert np.array_equal(rx.imag[4:], x.imag[:-4]) and np.array_equal(rx.real[4:], x.real[:-4]) # both rails 4 samples late, aligned

def test_stagger_any_block_cut():
    rng = np.random.default_rng(2)
    x = (rng.standard_normal(97) + 1j*rng.standard_normal(97)).astype(np.complex64)
    ref = stream(x, [97], rp.wave.Stagger(), Fs=8.0, sps=8)
    for cuts in CUTS(97):
        assert np.array_equal(stream(x, cuts, rp.wave.Stagger(), Fs=8.0, sps=8), ref)

def test_stagger_needs_whole_samples():
    with pytest.raises(ValueError):
        stream(np.zeros(16, np.complex64), [16], rp.wave.Stagger(), Fs=7.0, sps=7)
    with pytest.raises(ValueError):
        stream(np.zeros(16, np.float32), [16], rp.wave.Stagger(), Fs=8.0, sps=8)

def oqpsk(syms, sps):
    d = rp.Data(syms.astype(np.complex64), Fs=1e6, sps=1)
    return d + rp.signal.sample.up.Zeros(sps) + rp.signal.filters.load("pulse", "halfsine/halfsine") + rp.wave.Stagger()

def test_oqpsk_has_constant_envelope_and_no_phase_jumps():
    """Half-sine pulses on staggered rails: |s| is constant (MSK), so the phase moves by at most pi/2 per half symbol"""
    rng = np.random.default_rng(3)
    sym = (rng.choice([-1, 1], 200) + 1j*rng.choice([-1, 1], 200))/np.sqrt(2)
    w = oqpsk(sym, 8).data
    mid = w[40:-40]
    assert np.allclose(np.abs(mid)**2, 0.5, atol=1e-5)

def test_oqpsk_coherent_loopback_symbols():
    rng = np.random.default_rng(4)
    sps = 8
    sym = (rng.choice([-1, 1], 300) + 1j*rng.choice([-1, 1], 300))/np.sqrt(2)
    tx = oqpsk(sym, sps)
    rx = tx + rp.wave.Stagger(rail="i") + rp.signal.filters.load("pulse", "halfsine/halfsine") + rp.signal.sample.down.Decimate(sps, offset=sps + sps//2)
    got = rx.data*2/sps # the matched filter's gain at the peak is sps/2
    n = min(len(got), len(sym))
    assert n > 250
    assert np.allclose(got[:n], sym[:n], atol=1e-5)

# ---- FM

def levels_to_fm(levels, sps, h, pulse, step=2.0, blocks=None):
    lv = np.asarray(levels, np.float32)
    d = rp.Data(lv + 0j, Fs=sps*1e3, sps=1) + rp.signal.sample.up.Zeros(sps)
    w = d + pulse + rp.wave.FM(h, step)
    return w

def test_fm_is_constant_envelope_and_phase_continuous_across_blocks():
    rng = np.random.default_rng(5)
    x = rng.choice([-1.0, 1.0], 160).astype(np.float32)
    ref = stream(x, [160], rp.wave.FM(0.5), Fs=8.0, sps=8)
    assert np.allclose(np.abs(ref), 1, atol=1e-6)
    for cuts in CUTS(160):
        assert np.allclose(stream(x, cuts, rp.wave.FM(0.5), Fs=8.0, sps=8), ref, atol=1e-5)

def test_fm_tone_spacing():
    """A held level a is a tone of a*h/step cycles per symbol: levels +-1 at h = 0.5 are +-0.25 cycles per symbol"""
    sps = 16
    out = stream(np.ones(160, np.float32), [160], rp.wave.FM(0.5), Fs=float(sps), sps=sps)
    dphi = np.angle(out[1:]*np.conj(out[:-1]))
    assert np.allclose(dphi*sps/(2*np.pi), 0.25, atol=1e-5)

@pytest.mark.parametrize("M,h", [(2, 0.5), (2, 1.0), (4, 1/3)])
def test_fsk_rect_loopback(M, h):
    rng = np.random.default_rng(6)
    sps = 8
    lev = (2*rng.integers(0, M, 200) - (M - 1)).astype(np.float32)
    w = levels_to_fm(lev, sps, h, shaping.ShapingRect(sps, sps))
    dem = w + rp.wave.FMDemod(rp.wave.fm_gain(h, sps)) + rp.signal.sample.down.Decimate(sps, offset=sps//2)
    assert np.allclose(dem.data.real if dem.data.dtype.kind == "c" else dem.data, lev[:len(dem.data)], atol=1e-4)

def test_gfsk_loopback():
    """BLE-like: BT 0.5, h 0.5. Some sampling phase recovers every symbol (the pulse's delay is not a whole number of samples)"""
    rng = np.random.default_rng(7)
    sps = 8
    lev = rng.choice([-1.0, 1.0], 300).astype(np.float32)
    w = levels_to_fm(lev, sps, 0.5, rp.signal.filters.load("pulse", "gaussian/bt0p5"))
    freq = (w + rp.wave.FMDemod(rp.wave.fm_gain(0.5, sps))).data
    found = False
    for off in range(sps):
        pick = freq[off::sps]
        n = min(len(pick), len(lev))
        pick, ref = pick[8:n - 8], lev[8 - 0:n - 8]
        for lag in range(0, 8):
            m = min(len(pick) - lag, len(ref))
            if m > 200 and np.all(np.sign(pick[lag:lag + m]) == ref[:m]):
                found = True
    assert found

# ---- FSK from the layers: levels (a constellation) + pulse + FM, and back

def fsk_tx(c, bits, sps, h, pulse=None):
    syms = rp.Data(bits, Fs=sps*1e3, sps=1) * c
    pulse = shaping.ShapingRect(sps, sps) if pulse is None else pulse
    return syms + rp.signal.sample.up.Zeros(sps) + pulse + rp.wave.FM(h, c.min_distance)

def fsk_rx(wave, c, sps, h):
    """Discriminator, integrate over the symbol, one level per symbol (the first has no sample before it)"""
    lv = wave + rp.wave.FMDemod(rp.wave.fm_gain(h, sps, c.min_distance)) + shaping.ShapingRect(sps, sps, 1/sps) + rp.signal.sample.down.Decimate(sps, offset=sps - 1)
    return lv.data

@pytest.mark.parametrize("M", [2, 4])
def test_fsk_round_trip_and_noise(M):
    rng = np.random.default_rng(8)
    c, sps, h = rp.symbol.ASK(M), 8, 1.0
    bits = rng.integers(0, 2, c.bits_per_symbol*500).astype(bool)
    wave = fsk_tx(c, bits, sps, h)
    assert wave.sps == sps and len(wave) == 500*sps and np.allclose(np.abs(wave.data), 1.0, atol=1e-5)
    k = c.bits_per_symbol
    got = fsk_rx(wave, c, sps, h)
    assert np.array_equal(c.llr(got[1:].astype(np.complex64)) < 0, bits[k:len(got)*k])
    noisy = rp.Data(wave.data + 0.3*(rng.normal(size=len(wave)) + 1j*rng.normal(size=len(wave))).astype(np.complex64), Fs=sps*1e3, sps=sps)
    got = fsk_rx(noisy, c, sps, h)
    assert np.mean((c.llr(got[1:].astype(np.complex64)) < 0) != bits[k:len(got)*k]) < 0.05

def test_fsk_transmit_in_chunks_matches_one_shot():
    rng = np.random.default_rng(9)
    c, sps = rp.symbol.ASK(2), 8
    levels = np.real(c.modulate(rng.integers(0, 2, 400).astype(bool))).astype(np.complex64)
    pulse = shaping.ShapingRect(sps, sps)
    def chain(x, cuts):
        up, fm = rp.signal.sample.up.Zeros(sps), rp.wave.FM(0.5, c.min_distance) # one of each: their state lives in the stream, per stage
        s, outs, i = None, [], 0
        for b in cuts:
            d = rp.Data(x[i:i + b], stream=s, Fs=sps*1e3, sps=1)
            s = d.stream
            outs.append((d + up + pulse + fm).data)
            i += b
        return np.concatenate(outs)
    ref = chain(levels, [400])
    for cuts in ([301, 99], [1]*40 + [360], [0, 399, 1]):
        assert np.allclose(chain(levels, cuts), ref, atol=1e-5)
