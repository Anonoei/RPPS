"""The filter, demodulator loop, equalizer and resampler kernels give the Python reference's output (off and on)"""
import numpy as np
import pytest

import rpps as rp
from rpps import _accel
from rpps.signal.sample.fractional import design_bank, read_python
from support.native import both
from support.signals import noise

# ---------------------------------------------------------------- stages: IIR, demod, equalizers
def stage_run(make, x, cut, **rates):
    stage, stream, outs, i = make(), None, [], 0
    for b in cut:
        d = rp.Data(x[i:i+b], stream=stream, **rates)
        stream = d.stream
        outs.append(np.asarray((d + stage).data))
        i += b
    return np.concatenate(outs)

CUT = [1]*20 + [3000, 777, 20000 - 3797]

@pytest.mark.parametrize("kind", ["real", "complex"])
def test_iir(kind, rng):
    x = rng.standard_normal(20000) + (1j*rng.standard_normal(20000) if kind == "complex" else 0)
    x = x.astype(np.complex64 if kind == "complex" else np.float32)
    make = lambda: rp.signal.filters.iir.IIR(rp.signal.filters.iir.design("butter", 6, "lowpass", 0.1))
    ref, got = both(lambda: stage_run(make, x, CUT, Fs=1e3))
    assert np.allclose(ref, got, rtol=1e-5, atol=1e-6)

@pytest.mark.parametrize("make", [
    lambda: rp.wave.AMDemod(1e-3), lambda: rp.sync.Squelch(-10.0, 3.0, 0.01),
    lambda: rp.signal.DCBlock(0.999), lambda: rp.signal.Deemphasis(75e-6)], ids=["am", "squelch", "dcblock", "deemph"])
@pytest.mark.parametrize("kind", ["real", "complex"])
def test_demod_loops(make, kind, rng):
    env = np.where((np.arange(20000) // 4000) % 2 == 0, 0.05, 2.0) # bursts well clear of the squelch threshold
    x = env*(rng.standard_normal(20000) + (1j*rng.standard_normal(20000) if kind == "complex" else 0)) + 0.3
    x = x.astype(np.complex64 if kind == "complex" else np.float32)
    ref, got = both(lambda: stage_run(make, x, CUT, Fs=48e3))
    assert ref.dtype == got.dtype
    assert np.allclose(ref, got, rtol=1e-4, atol=1e-4)

def isi_qpsk(rng, n=6000):
    sym = (rng.choice([-1, 1], n) + 1j*rng.choice([-1, 1], n))/np.sqrt(2)
    y = np.convolve(sym, [1.0, 0.35 - 0.2j, 0.1j])[:n] + 0.02*(rng.standard_normal(n) + 1j*rng.standard_normal(n))
    return sym.astype(np.complex64), y.astype(np.complex64)

@pytest.mark.parametrize("which", ["cma", "lms-dd", "lms-training"])
def test_equalizers(which, rng):
    sym, y = isi_qpsk(rng)
    pts = (np.array([1+1j, 1-1j, -1+1j, -1-1j])/np.sqrt(2)).astype(np.complex64)
    make = {"cma": lambda: rp.sync.loops.CMA(11, 0.002),
            "lms-dd": lambda: rp.sync.loops.LMS(11, 0.005, points=pts),
            "lms-training": lambda: rp.sync.loops.LMS(11, 0.005, points=pts, training=np.roll(sym, 5)[:2000])}[which]
    cut = [1]*10 + [1000, 4990]
    ref, got = both(lambda: stage_run(make, y, cut, Fs=1e3, sps=1))
    assert len(ref) == len(got) > 5900
    assert np.allclose(ref, got, atol=1e-3)

@pytest.mark.parametrize("taps", [2, 6, 10, 16, 24, 32, 34, 64])
@pytest.mark.parametrize("dtype", [np.complex64, np.float32])
def test_native_read_matches_the_python_reference(dtype, taps):
    """Every tap count: the vector loops, their 8- and 16-wide steps and the scalar tail"""
    table = design_bank(taps, 512, 0.185)
    buf = noise(50000, dtype, 4)
    step = 20e6/10.7622e6
    n = int((len(buf) - 1 - taps//2 - (taps//2 - 1))/step) + 1
    want = read_python(buf, table, taps/2 - 1.0, step, n)
    got = _accel.native.polyphase_read(buf, table.astype(np.float32), taps/2 - 1.0, step, n, 0)
    assert got.dtype == dtype and got.shape == want.shape
    np.testing.assert_allclose(got, want, atol=3e-6)

def test_native_rotate_matches_numpy():
    x = noise(100000)
    out = np.empty_like(x)
    w = 2*np.pi*-0.5e6/20e6
    ph = _accel.native.rotate_into(x, out, 1.3, w, 0)
    np.testing.assert_allclose(out, x*np.exp(1j*(1.3 + w*np.arange(len(x)))), atol=2e-6)
    assert ph == pytest.approx((1.3 + w*len(x)) % (2*np.pi))
