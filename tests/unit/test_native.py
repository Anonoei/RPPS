"""The C++ kernels (rpps._native) give the Python loops' output, across any cut into blocks

Skipped when the extension is not built (`python setup.py build_ext -i`).
"""
import numpy as np
import pytest

import rpps as rp
from rpps import _accel

pytestmark = pytest.mark.skipif(not _accel.available(), reason="rpps._native is not built")

class Counting:
    """rpps._native, counting the kernel calls (a native run that silently fell back to Python would compare equal)"""
    def __init__(self, mod):
        self.mod, self.calls = mod, 0
    def __getattr__(self, name):
        self.calls += 1
        return getattr(self.mod, name)

def run(make, x, blocks, native, **rates):
    """x cut into `blocks` through a fresh stage, with the kernels on or off"""
    saved = _accel.native
    spy = Counting(saved)
    _accel.native = spy if native else None
    try:
        stage = make()
        stream, outs, i = None, [], 0
        for b in blocks:
            d = rp.Data(x[i:i+b], stream=stream, **rates)
            stream = d.stream
            outs.append((d + stage).data)
            i += b
        if native:
            assert spy.calls > 0, "the native kernel never ran"
        return np.concatenate(outs), stream.states[stage]
    finally:
        _accel.native = saved

def psk(rng, n, M, cfo=0.0, snr_db=20.0):
    sym = np.exp(1j*(2*np.pi*rng.integers(0, M, n)/M + (np.pi/4 if M == 4 else 0.0)))
    sym = sym*np.exp(1j*(2*np.pi*cfo*np.arange(n) + 0.7))
    noise = (rng.standard_normal(n) + 1j*rng.standard_normal(n))*10**(-snr_db/20)/np.sqrt(2)
    return (sym + noise).astype(np.complex64)

def shaped(rng, n_sym, sps, delay):
    """QPSK, rectangular pulses smoothed to a triangle, delayed by `delay` samples"""
    sym = rng.choice([-1, 1], n_sym) + 1j*rng.choice([-1, 1], n_sym)
    sig = np.convolve(np.repeat(sym, sps), np.ones(sps)/sps, mode="same")
    sig = np.roll(sig, delay) + 0.05*(rng.standard_normal(len(sig)) + 1j*rng.standard_normal(len(sig)))
    return sig.astype(np.complex64)

CUTS = ([20000], [4096]*4 + [20000 - 4*4096], [1]*30 + [20000 - 30])

@pytest.mark.parametrize("order", [2, 4, 8])
@pytest.mark.parametrize("cut", range(len(CUTS)))
def test_costas_matches_python(order, cut):
    x = psk(np.random.default_rng(order), 20000, order, cfo=2e-3)
    make = lambda: rp.sync.Costas.from_bandwidth(0.02, order=order)
    ref, st_ref = run(make, x, [len(x)], native=False, Fs=1e3, sps=1)
    y, st = run(make, x, CUTS[cut], native=True, Fs=1e3, sps=1)
    assert y.dtype == ref.dtype == np.complex64
    assert np.allclose(y, ref, atol=1e-4)
    assert st.freq == pytest.approx(st_ref.freq, abs=1e-9)
    assert abs(st.freq - 2*np.pi*2e-3) < 2e-3 # it locked to the offset

def test_costas_with_a_replaced_error_function_runs_in_python():
    x = psk(np.random.default_rng(0), 500, 2)
    c = rp.sync.Costas(0.05, 0.001, 2)
    c.err_func = lambda s: 0.0 # an open loop: the phase never moves
    y = (rp.Data(x, Fs=1e3, sps=1) + c).data
    assert np.allclose(y, x)

@pytest.mark.parametrize("log", [True, False])
@pytest.mark.parametrize("kind", [np.complex64, np.float32, np.complex128])
def test_agc_matches_python(log, kind):
    rng = np.random.default_rng(1)
    env = np.concatenate((np.full(7000, 0.1), np.full(7000, 3.0), np.full(6000, 0.5)))
    x = env*(rng.standard_normal(20000) + (1j*rng.standard_normal(20000) if np.dtype(kind).kind == "c" else 0))
    x = x.astype(kind)
    make = lambda: rp.sync.AGC(1.0, 0.005, log=log, max_gain=50.0)
    ref, st_ref = run(make, x, [len(x)], native=False, Fs=1e3)
    for cut in CUTS[1:]:
        y, st = run(make, x, cut, native=True, Fs=1e3)
        assert y.dtype == ref.dtype
        assert np.allclose(y, ref, rtol=1e-4, atol=1e-5)
        assert float(st.gain) == pytest.approx(float(st_ref.gain), rel=1e-4)

@pytest.mark.parametrize("ted, biphase", [("mm", False), ("gardner", False), ("mm", True)])
def test_mm_matches_python(ted, biphase):
    sps = 8
    x = shaped(np.random.default_rng(5), 2500, sps, delay=3)
    make = lambda: rp.sync.time.MM(alpha=0.1, ted=ted, biphase=biphase)
    for cut in ([len(x)], [4000, 4000, len(x) - 8000], [1]*40 + [len(x) - 40]):
        ref, st_ref = run(make, x, cut, native=False, Fs=8e3, sps=sps)
        y, st = run(make, x, cut, native=True, Fs=8e3, sps=sps)
        assert len(y) == len(ref) > 2000 and st.count == st_ref.count and st.pos == st_ref.pos # the same walk
        assert np.allclose(y, ref, atol=1e-5)
        assert float(st.est) == pytest.approx(float(st_ref.est), abs=1e-5)

def test_kernels_release_the_gil():
    """Python keeps running while a kernel runs in another thread. Holding the GIL, the kernel would stop this thread for
    its whole run (but for switch intervals, 0.1 ms here): well under 1% of the counting rate it has alone. Released, it
    keeps a large share even with every core busy (pytest -n), so the bound is 5%: no wall-clock race between threads"""
    import sys, threading, time
    x = psk(np.random.default_rng(2), 2_000_000, 4)
    work = lambda: _accel.native.costas(x, 0.0, 0.0, 0.01, 1e-4, 4, 0.0)
    def count(until):
        n = 0
        while not until():
            n += 1
        return n
    t0 = time.perf_counter()
    alone = count(lambda: time.perf_counter() - t0 > 0.05)/(time.perf_counter() - t0) # iterations a second
    started, done = threading.Event(), threading.Event()
    def run():
        started.set()
        work()
        done.set()
    old = sys.getswitchinterval()
    sys.setswitchinterval(1e-4)
    try:
        th = threading.Thread(target=run)
        th.start()
        started.wait()
        t = time.perf_counter()
        n = count(done.is_set)
        during = time.perf_counter() - t
        th.join()
    finally:
        sys.setswitchinterval(old)
    assert during > 0.02 and n > 0.05*alone*during

@pytest.mark.parametrize("taps", [1, 33, 1025])
@pytest.mark.parametrize("cut", [[3000], [1, 2047, 7, 900]]) # short blocks: where the direct (native) path runs
def test_fir_stream_matches_python(taps, cut):
    rng = np.random.default_rng(3)
    x = (rng.standard_normal(sum(cut)) + 1j*rng.standard_normal(sum(cut))).astype(np.complex64)
    h = rp.signal.filters.designer.designer(taps, 0.2) if taps > 1 else np.ones(1)
    make = lambda: rp.signal.filters.filters.ConvFilter(h)
    nat, _ = run(make, x, cut, True, Fs=1.0)
    py, _ = run(make, x, cut, False, Fs=1.0)
    ref = np.convolve(x, h)[:len(x)]
    assert nat.dtype == np.complex64
    np.testing.assert_allclose(nat, py, atol=1e-4*np.sqrt(taps))
    np.testing.assert_allclose(nat, ref, atol=1e-4*np.sqrt(taps))

def test_fft_convolution_stays_single_and_close():
    rng = np.random.default_rng(4)
    x = (rng.standard_normal(1 << 16) + 1j*rng.standard_normal(1 << 16)).astype(np.complex64)
    h = rp.signal.filters.designer.designer(257, 0.1).astype(np.float32)
    from rpps.signal.filters.convolve import fft_pays
    assert fft_pays(len(x), len(h))
    y = rp.signal.filters.convolve(x, h)
    assert y.dtype == np.complex64
    np.testing.assert_allclose(y, np.convolve(x.astype(np.complex128), h.astype(np.float64)), atol=1e-4)

@pytest.mark.parametrize("cut", [[5000], [1, 4095, 7, 900]])
def test_shift_freq_matches_python(cut):
    rng = np.random.default_rng(5)
    x = (rng.standard_normal(sum(cut)) + 1j*rng.standard_normal(sum(cut))).astype(np.complex64)
    make = lambda: rp.signal.filters.ShiftFreq(1234.5)
    nat, st_n = run(make, x, cut, True, Fs=1e5)
    py, st_p = run(make, x, cut, False, Fs=1e5)
    assert nat.dtype == py.dtype == np.complex64
    np.testing.assert_allclose(nat, py, atol=1e-5)
    assert st_n.phase == pytest.approx(st_p.phase, abs=1e-9)
