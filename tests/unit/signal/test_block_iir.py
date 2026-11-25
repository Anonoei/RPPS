"""The recursive filters with no loop over samples (rpps.signal.filters.iir: one state space system, chunks as matrix
products, the chunk states by a log-depth scan) against the sample-by-sample recursions they replace"""
from contextlib import nullcontext

import numpy as np
import pytest

import rpps as rp
from rpps import _accel
from support.blocks import run
from support.native import kernels

CUTS = ([20000], [1]*40 + [19960], [777]*25 + [20000 - 777*25], [5000, 0, 15000])

# ---- the reference recursions (what the stages ran, one sample at a time), in double precision

def _double(x):
    return np.asarray(x, dtype=np.complex128 if np.iscomplexobj(x) else np.float64)

def ref_sos(sos, x):
    """Transposed direct form II, section after section, double precision"""
    y = np.asarray(x, dtype=np.complex128 if np.iscomplexobj(x) else np.float64).copy()
    for b0, b1, b2, _, a1, a2 in rp.signal.filters.iir.as_sos(sos):
        z1 = z2 = 0.0
        out = np.empty_like(y)
        for i, v in enumerate(y):
            out[i] = b0*v + z1
            z1, z2 = -a1*z1 + z2 + (b1 - a1*b0)*v, -a2*z1 + (b2 - a2*b0)*v
        y = out
    return y

def ref_dcblock(x, r):
    x = _double(x)
    out, x1, y1 = np.empty(len(x), np.result_type(x, np.float64)), 0.0, 0.0
    for i, v in enumerate(x):
        y1 = v - x1 + r*y1
        x1 = v
        out[i] = y1
    return out

def ref_deemphasis(x, tau, Fs):
    x = _double(x)
    a, y, out = 1 - np.exp(-1/(tau*Fs)), 0.0, np.empty(len(x), np.result_type(x, np.float64))
    for i, v in enumerate(x):
        y += a*(v - y)
        out[i] = y
    return out

def ref_am(x, alpha):
    env, mean, out = np.abs(x).astype(np.float64), None, np.empty(len(x))
    for i, v in enumerate(env):
        mean = v if mean is None else mean + alpha*(v - mean)
        out[i] = v - mean
    return out

def ref_squelch(x, open_db, hyst_db, alpha):
    x = _double(x)
    lo, hi = 10**((open_db - hyst_db)/10), 10**(open_db/10)
    lvl, op, out = None, False, np.zeros_like(x)
    for i, p in enumerate(np.abs(x).astype(np.float64)**2):
        lvl = p if lvl is None else lvl + alpha*(p - lvl)
        op = lvl > (lo if op else hi)
        if op:
            out[i] = x[i]
    return out

@pytest.fixture(params=["native", "block"])
def path(request):
    """The C++ kernel (when built) and the block form (numpy; the loops' Python loop) both"""
    if request.param == "native" and _accel.native is None:
        pytest.skip("rpps._native is not built")
    with nullcontext() if request.param == "native" else kernels(False):
        yield request.param

@pytest.fixture
def signals(rng):
    env = np.where((np.arange(20000)//4000) % 2 == 0, 0.05, 2.0) # bursts well clear of the squelch threshold
    real = env*rng.standard_normal(20000) + 0.3
    return {"real": real, "complex": real + 1j*env*rng.standard_normal(20000)}

@pytest.mark.parametrize("sos", [rp.signal.filters.iir.design("butter", 8, "lowpass", 0.02),
                                 rp.signal.filters.iir.design("cheby1", 6, "bandpass", (0.1, 0.2), ripple_db=1.0),
                                 rp.signal.filters.iir.biquad("notch", 0.1, q=8.0),
                                 [[1.0, 0.0, 0.0, 1.0, -1.0, 0.0]]], # an integrator: a pole on the unit circle
                         ids=["butter8", "cheby1-bp", "notch", "integrator"])
@pytest.mark.parametrize("kind", ["real", "complex"])
def test_iir_is_the_recursion_for_any_cut(sos, kind, signals, path):
    x = signals[kind]
    f = rp.signal.filters.IIR(sos)
    want = ref_sos(sos, x)
    scale = np.max(np.abs(want))
    assert np.max(np.abs(f.run(x) - want)) <= 1e-11*scale # double in, double out: to rounding
    for cut in CUTS: # a stream (its samples single precision): to single precision
        got = run(lambda: rp.signal.filters.IIR(sos), x, cut, Fs=48e3)
        want = ref_sos(sos, x.astype(np.complex64 if kind == "complex" else np.float32))
        assert np.max(np.abs(got - want)) <= 1e-6*scale, (path, cut[:2])

@pytest.mark.parametrize("kind", ["real", "complex"])
def test_first_order_stages_are_their_recursions(kind, signals, path):
    x = np.asarray(rp.Data(signals[kind]).data) # (what a stream holds: complex is single precision)
    cases = [
        (lambda: rp.signal.DCBlock(0.999), ref_dcblock(x, 0.999)),
        (lambda: rp.signal.Deemphasis(75e-6), ref_deemphasis(x, 75e-6, 48e3)),
        (lambda: rp.wave.AMDemod(1e-3), ref_am(x, 1e-3)),
        (lambda: rp.sync.Squelch(-10.0, 3.0, 0.01), ref_squelch(x, -10.0, 3.0, 0.01)),
        # the second-class loops (their kernel, or their Python loop): the same recursions sample by sample
        (lambda: rp.signal.loops.DCBlock(0.999), ref_dcblock(x, 0.999)),
        (lambda: rp.signal.loops.Deemphasis(75e-6), ref_deemphasis(x, 75e-6, 48e3)),
        (lambda: rp.wave.loops.AMDemod(1e-3), ref_am(x, 1e-3)),
        (lambda: rp.sync.loops.Squelch(-10.0, 3.0, 0.01), ref_squelch(x, -10.0, 3.0, 0.01)),
    ]
    for make, want in cases:
        for cut in CUTS:
            got = run(make, x, cut, Fs=48e3)
            assert got.shape == want.shape
            tol = 1e-9 if got.dtype in (np.float64, np.complex128) else 2e-6 # the output's own precision
            assert np.allclose(got, want, rtol=tol, atol=tol), (str(make()), path, cut[:2])

def test_squelch_gate_holds_between_the_thresholds():
    gate = rp.sync.squelch.gate
    level = np.array([0.0, 2.0, 1.5, 1.5, 0.5, 1.5, 3.0])
    assert gate(level, 1.0, 1.8).tolist() == [False, True, True, True, False, False, True]
    assert gate(np.array([1.5, 1.5]), 1.0, 1.8, was_open=True).tolist() == [True, True]
    assert gate(np.zeros(0), 1.0, 1.8).tolist() == []

def test_single_precision_in_single_precision_out(signals, path):
    x = signals["complex"].astype(np.complex64)
    assert run(lambda: rp.signal.filters.IIR(rp.signal.filters.iir.single_pole(0.1)), x, [20000], Fs=48e3).dtype == np.complex64
    assert run(lambda: rp.signal.DCBlock(), x, [20000], Fs=48e3).dtype == np.complex64
