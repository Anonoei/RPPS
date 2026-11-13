"""Differential phase coding of symbols"""
import numpy as np
import pytest

import rpps as rp
from rpps.symbol import Differential

def hard(d):
    return np.asarray(d.data) < 0

def symbols_like(sym, ctx=None):
    d = rp.Data(b"\x00")
    return d.derive(np.asarray(sym, np.complex64), rp.Category.SYMBOL, rp.dtype.SYMBOLS, rp.base.Mod, {} if ctx is None else ctx)

@pytest.mark.parametrize("name", ["bpsk", "qpsk", "qpsk_axes", "8psk"])
def test_round_trip(name):
    rng = np.random.default_rng(0)
    c, diff = rp.symbol.load(name), Differential()
    bits = rng.integers(0, 2, c.bits_per_symbol*300).astype(bool)
    sym = rp.Data(bits) * c * diff
    assert np.allclose(np.abs(sym.data), 1, atol=1e-6)
    assert np.array_equal(hard(sym / diff / c), bits)

def test_dbpsk_is_a_phase_flip_per_one():
    diff, c = Differential(), rp.symbol.load("bpsk")
    bits = rp.Data(np.packbits(np.array([0, 1, 1, 0, 1, 0, 0, 0], dtype=np.uint8)))
    s = (bits * c * diff).data
    flips = np.sign((s[1:]*np.conj(s[:-1])).real) < 0
    assert flips.tolist() == [True, True, False, True, False, False, False]

def test_pi_over_4_dqpsk_moves_by_odd_multiples_of_45_degrees():
    rng = np.random.default_rng(1)
    c, diff = rp.symbol.load("qpsk"), Differential() # the points at 45, 135, 225, 315 degrees are the phase changes
    sym = (rp.Data(rng.integers(0, 256, 50).astype(np.uint8)) * c * diff).data
    step = np.degrees(np.angle(sym[1:]*np.conj(sym[:-1]))) % 360
    assert set(np.round(step).astype(int)) <= {45, 135, 225, 315}

def test_constant_phase_and_gain_cancel_after_the_first_symbol():
    rng = np.random.default_rng(2)
    c, diff = rp.symbol.load("qpsk"), Differential()
    bits = rng.integers(0, 2, 2*400).astype(bool)
    tx = rp.Data(bits) * c * diff
    rx = symbols_like(0.37*np.exp(1j*1.9)*tx.data)
    got = hard(rx / diff / c)
    assert np.array_equal(got[2:], bits[2:]) # the first symbol's reference is the phase 0 the transmitter started from

def test_joining_mid_stream_is_right_from_the_second_symbol():
    rng = np.random.default_rng(3)
    c, diff = rp.symbol.load("bpsk"), Differential()
    bits = rng.integers(0, 2, 200).astype(bool)
    tx = (rp.Data(bits) * c * diff).data
    got = hard(symbols_like(tx[57:]) / diff / c)
    assert np.array_equal(got[1:], bits[58:])

def test_any_block_cut():
    rng = np.random.default_rng(4)
    c, diff = rp.symbol.load("qpsk_axes"), Differential()
    bits = rng.integers(0, 2, 2*257).astype(bool)
    sym = (rp.Data(bits) * c).data
    ref = (symbols_like(sym) * diff).data
    for cuts in ([257], [100, 100, 57], [1]*30 + [227], [0, 5, 252]):
        ctx, outs, i = None, [], 0
        for b in cuts:
            r = symbols_like(sym[i:i + b], ctx) * diff
            ctx = dict(r.ctx)
            outs.append(r.data)
            i += b
        assert np.allclose(np.concatenate(outs), ref, atol=1e-5)
        ctx, outs, i = None, [], 0
        for b in cuts:
            r = symbols_like(ref[i:i + b], ctx) / diff
            ctx = dict(r.ctx)
            outs.append(r.data)
            i += b
        assert np.allclose(np.concatenate(outs), sym, atol=1e-5)
