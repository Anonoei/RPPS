"""Smoke tests for the small helpers with no caller in the package, so they keep working"""
from types import SimpleNamespace

import numpy as np
import pytest

import rpps as rp
from rpps.code import blocker


def test_agc_lin_is_linear_agc():
    a = rp.sync.agc.AGC_lin(2.0, 0.05)
    x = (0.1*np.exp(1j*np.linspace(0, 50, 20000))).astype(np.complex64)
    y = rp.Data(x, Fs=1e6) + a
    assert np.mean(np.abs(y.data[-2000:])) == pytest.approx(2.0, rel=0.05)

def test_bits_base2_base10_round_trip():
    rows = rp.math.bits.to_base2(4)
    assert rows.shape == (16, 4)
    assert (rp.math.bits.to_base10(rows) == np.arange(16)).all()

def test_hamming_dist():
    assert blocker.hamming_dist(np.array([1, 0, 1, 1]), np.array([0, 0, 1, 0])) == 2

def test_parity_block_round_trip():
    P = np.array([[1, 1, 0], [0, 1, 1], [1, 1, 1], [1, 0, 1]])
    G, H = rp.math.gf2.parity_to_gh(7, 4, P)
    assert (rp.math.gf2.gh_to_parity(7, 4, G=G) == P).all()
    assert (rp.math.gf2.gh_to_parity(7, 4, H=H) == P).all()

def test_entropy():
    assert rp.math.bits.entropy(np.array([0, 1, 1, 0])) == pytest.approx(1.0)
    assert rp.math.bits.entropy(np.zeros(10, dtype=int)) == 0.0

def test_sos2zpk_round_trip():
    sos = rp.signal.filters.iir.design("butter", 4, "lowpass", 0.2)
    z, p, k = rp.signal.filters.iir.sos2zpk(sos)
    assert len(p) == 4 and np.all(np.abs(p) < 1)
    assert k != 0

def test_high_pass_kaiser_blocks_dc():
    f = rp.signal.filters.high_pass_kaiser(0.25, 0.05)
    assert abs(np.sum(f.filt)) < 1e-2

def test_find_rate():
    rng = np.random.default_rng(0)
    sps = 8
    syms = np.exp(1j*np.pi/4*(2*rng.integers(0, 4, 4000) + 1))
    x = np.convolve(np.repeat(syms, sps), np.hanning(sps), "same")
    assert rp.analysis.find_rate(x) == pytest.approx(1/sps, rel=0.01)

def test_find_sync_and_decode_conv():
    rng = np.random.default_rng(1)
    B = rng.integers(0, 256, 204*8, dtype=np.uint8)
    B[::204] = 0x47
    bits = np.concatenate((rng.integers(0, 2, 3, dtype=np.uint8), np.unpackbits(B)))
    frac, off, phase = rp.analysis.framing.bits.find_sync(bits)
    assert frac == 1.0 and off == 3 and phase == 0
    code = rp.code.load("conv/k7_133_171/1_2")
    msg = rng.integers(0, 2, 400).astype(bool)
    hyp = SimpleNamespace(coding=("conv", "k7_133_171/1_2"), params={"offset": 0})
    out = rp.analysis.code.conv.mother.decode_conv(code.encoder.encode(msg).astype(np.uint8), hyp)
    assert (np.asarray(out[:300]).astype(bool) == msg[:300]).all()

def test_shaping_mat_rrc_shapes_a_block():
    f = rp.signal.filters.ShapingMatRRC(16, 4)
    assert f.pulse.shape == (16, 64)
    assert f.burst(np.ones(40)).shape == (160,)
