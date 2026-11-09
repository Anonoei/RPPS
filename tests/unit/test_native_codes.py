"""The C++ kernels of the decoders, filters and equalizers give the Python reference's output

Each test runs the same input with rpps._native off and on (and checks the kernel ran). Skipped when it isn't built.
"""
import numpy as np
import pytest

import rpps as rp
from rpps import _accel

pytestmark = pytest.mark.skipif(not _accel.available(), reason="rpps._native is not built")

class Counting:
    def __init__(self, mod):
        self.mod, self.calls = mod, 0
    def __getattr__(self, name):
        self.calls += 1
        return getattr(self.mod, name)

def both(fn):
    """fn() with the Python loops, then with the kernels: (python, native)"""
    saved = _accel.native
    try:
        _accel.native = None
        ref = fn()
        spy = Counting(saved)
        _accel.native = spy
        got = fn()
        assert spy.calls > 0, "the native kernel never ran"
    finally:
        _accel.native = saved
    return ref, got

def noisy_llr(bits, rng, ebn0=2.0):
    s = 1.0 - 2.0*np.asarray(bits, dtype=np.float64)
    sigma = 10**(-ebn0/20)
    return 2*(s + sigma*rng.standard_normal(len(s)))/sigma**2

# ---------------------------------------------------------------- Viterbi (conv) and fsm
@pytest.mark.parametrize("name", ["k7_133_171/1_2", "k7_171_133/3_4", "k3_5_7/1_2"])
@pytest.mark.parametrize("soft", [True, False])
def test_viterbi_frames(name, soft, rng):
    code = rp.coding.load("conv", name)
    msg = rng.integers(0, 2, 600).astype(bool)
    llr = noisy_llr(code._enc.encode(msg), rng, ebn0=6.0)
    x = llr if soft else (llr < 0)
    ref, got = both(lambda: code._dec.decode(x))
    assert np.array_equal(ref, got)
    n = min(len(got), len(msg)) - 10 # the end of a truncated code is unsettled
    assert np.mean(got[:n] != msg[:n]) < 0.02

def test_viterbi_stream_carries_state(rng):
    code = rp.coding.load("conv", "k7_133_171/1_2")
    llr = noisy_llr(code._enc.encode(rng.integers(0, 2, 2000).astype(bool)), rng)
    def stream():
        st, out = {}, []
        for i in range(0, len(llr), 302):
            part = llr[i:i+302]
            out.append(code._dec.decode(part, soft=True, state=st, final=i + 302 >= len(llr)))
        return np.concatenate(out)
    ref, got = both(stream)
    assert np.array_equal(ref, got)

def test_viterbi_tailbiting(rng):
    code = rp.coding.Conv([0o133, 0o171, 0o165], 7, termination="tailbiting")
    msg = rng.integers(0, 2, 40).astype(bool)
    llr = noisy_llr(code._enc.encode(msg), rng, ebn0=3.0)
    ref, got = both(lambda: code._dec.decode(llr))
    assert np.array_equal(ref, got) and np.array_equal(got, msg)

@pytest.mark.parametrize("name", ["trellis/p25_1_2", "trellis/atsc_8vsb"])
def test_fsm_batches(name, rng):
    code = rp.coding.load("block", name)._dec
    msg = rng.integers(0, 2, (12, 2*96)).astype(bool)
    enc = np.array([code.encode(m) for m in msg])
    if code.levels is not None: # amplitudes
        x = np.array([code.modulate(e) for e in enc]) + 0.6*rng.standard_normal((12, len(enc[0])//code.sym_bits))
    else:
        x = np.array([noisy_llr(e, rng, ebn0=3.0) for e in enc])
    ref, got = both(lambda: (code.decode_batch(x), code.errors))
    assert np.array_equal(ref[0], got[0])
    assert (ref[1] is None and got[1] is None) or np.array_equal(ref[1], got[1])

# ---------------------------------------------------------------- BCJR: conv, turbo, syndrome trellis
@pytest.mark.parametrize("maxlog", [False, True])
def test_bcjr_conv(maxlog, rng):
    code = rp.coding.load("conv", "k7_133_171/1_2", decoder="bcjr")
    code._dec.maxlog = maxlog
    llr = noisy_llr(code._enc.encode(rng.integers(0, 2, 300).astype(bool)), rng)
    ref, got = both(lambda: code._dec.app(llr, soft=True))
    assert np.allclose(ref, got, rtol=1e-9, atol=1e-9)

@pytest.mark.parametrize("maxlog", [True, False])
def test_turbo(maxlog, rng):
    code = rp.coding.load("block", "turbo/lte", K=104)
    code._dec.maxlog = maxlog
    msg = rng.integers(0, 2, 2*104).astype(bool)
    llr = noisy_llr(code._enc.encode(msg), rng, ebn0=0.5)
    ref, got = both(lambda: (code._dec.decode(llr), code._dec.llr.copy()))
    assert np.array_equal(ref[0], got[0])
    assert np.allclose(ref[1], got[1], rtol=1e-6, atol=1e-6)

def test_syndrome_trellis(rng):
    code = rp.coding.load("block", "hamming/15_11", decoder="trellis")
    words = code._enc.encode(rng.integers(0, 2, 11*20).astype(bool))
    llr = noisy_llr(words, rng, ebn0=4.0)
    ref, got = both(lambda: code._dec.decode(llr, soft=True))
    assert np.array_equal(ref, got)

# ---------------------------------------------------------------- LDPC
def test_ldpc_one_codeword_at_a_time(rng):
    code = rp.coding.load("block", "ldpc/gallager_96_48")
    k = code._dec.num # 50: the construction's H has dependent rows
    for _ in range(10):
        llr = noisy_llr(code._enc.encode(rng.integers(0, 2, k).astype(bool)), rng, ebn0=2.5)
        ref, got = both(lambda: (code._dec.decode(llr), code._dec.converged.copy()))
        assert np.array_equal(ref[0], got[0]) and np.array_equal(ref[1], got[1])

def test_ldpc_batch_is_each_codeword_on_its_own(rng):
    """The kernel stops each codeword when it converges: the numpy decoder run one codeword at a time"""
    code = rp.coding.load("block", "ldpc/gallager_96_48")
    k, n = code._dec.num, code._dec.den
    llr = noisy_llr(code._enc.encode(rng.integers(0, 2, k*40).astype(bool)), rng, ebn0=2.5)
    def each():
        out, conv = [], []
        for w in llr.reshape(40, n):
            out.append(code._dec.decode(w))
            conv.append(code._dec.converged[0])
        return np.array(out), np.array(conv)
    saved = _accel.native
    _accel.native = None
    try:
        ref = each()
    finally:
        _accel.native = saved
    got = code._dec.decode(llr).reshape(40, k), code._dec.converged
    assert np.array_equal(ref[0], got[0]) and np.array_equal(ref[1], got[1]) and got[1].sum() > 30

# ---------------------------------------------------------------- Reed-Solomon, BCH
def corrupt_symbols(code, msg_syms, rng, n_err, m=8):
    cw = code._enc.encode(msg_syms, symbols=True).reshape(-1, code._dec.den).copy()
    for w in cw:
        pos = rng.choice(len(w), n_err, replace=False)
        w[pos] ^= rng.integers(1, 1 << m, n_err)
    return cw

@pytest.mark.parametrize("name, n_err", [("rs/204_188", 8), ("rs/204_188", 9), ("rs/p25_24_12", 6), ("rs/p25p2_soemi", 2)])
def test_rs_errors(name, n_err, rng):
    code = rp.coding.load("block", name)
    k, m = code._dec.num, code._dec.m
    cw = corrupt_symbols(code, rng.integers(0, 1 << m, k*30), rng, n_err, m)
    ref, got = both(lambda: (code._dec.decode(cw.reshape(-1), symbols=True), code._dec.errors.copy()))
    assert np.array_equal(ref[0], got[0]) and np.array_equal(ref[1], got[1])

def test_rs_erasures_and_soft_retries(rng):
    code = rp.coding.load("block", "rs/204_188")
    msg = rng.integers(0, 256, 188*20)
    cw = corrupt_symbols(code, msg, rng, 0)
    erased = np.zeros(cw.shape, dtype=bool)
    for w, e in zip(cw, erased):
        pos = rng.choice(204, 12, replace=False)
        w[pos] ^= rng.integers(1, 256, 12)
        e[pos[:8]] = True # 8 of the 12 known: 8 erasures + 4 errors <= 16
    ref, got = both(lambda: (code._dec.decode(cw.reshape(-1), symbols=True, erasures=erased), code._dec.errors.copy()))
    assert np.array_equal(ref[0], got[0]) and np.array_equal(ref[1], got[1]) and (got[1] >= 0).all()
    # soft: 10 errors (beyond t = 8), the corrupted bytes least reliable
    cw = corrupt_symbols(code, msg, rng, 10)
    bits = code._dec.to_bits(cw.reshape(-1)).reshape(20, 204, 8)
    llr = np.where(bits, -4.0, 4.0) + 0.1*rng.standard_normal(bits.shape)
    clean = code._dec.to_bits(code._enc.encode(msg, symbols=True)).reshape(20, 204, 8)
    llr[bits != clean] = np.where(bits[bits != clean], -0.3, 0.3)
    ref, got = both(lambda: (code._dec.decode(llr.reshape(-1)), code._dec.errors.copy()))
    assert np.array_equal(ref[0], got[0]) and np.array_equal(ref[1], got[1])

def test_bch(rng):
    code = rp.coding.load("block", "bch/dvbs2_1_4")
    k, n = code._dec.num, code._dec.den
    cw = code._enc.encode(rng.integers(0, 2, k*3).astype(bool)).reshape(3, n).copy()
    for w, ne in zip(cw, (5, 12, 14)):
        w[rng.choice(n, ne, replace=False)] ^= True
    ref, got = both(lambda: (code._dec.decode(cw.reshape(-1)), code._dec.errors.copy()))
    assert np.array_equal(ref[0], got[0]) and np.array_equal(ref[1], got[1])
    assert list(got[1][:2]) == [5, 12]

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
    make = lambda: rp.filters.iir.IIR(rp.filters.iir.design("butter", 6, "lowpass", 0.1))
    ref, got = both(lambda: stage_run(make, x, CUT, Fs=1e3))
    assert np.allclose(ref, got, rtol=1e-5, atol=1e-6)

@pytest.mark.parametrize("make", [
    lambda: rp.demod.AMDemod(1e-3), lambda: rp.demod.Squelch(-10.0, 3.0, 0.01),
    lambda: rp.demod.DCBlock(0.999), lambda: rp.demod.Deemphasis(75e-6)], ids=["am", "squelch", "dcblock", "deemph"])
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
    make = {"cma": lambda: rp.equalize.CMA(11, 0.002),
            "lms-dd": lambda: rp.equalize.LMS(11, 0.005, points=pts),
            "lms-training": lambda: rp.equalize.LMS(11, 0.005, points=pts, training=np.roll(sym, 5)[:2000])}[which]
    cut = [1]*10 + [1000, 4990]
    ref, got = both(lambda: stage_run(make, y, cut, Fs=1e3, sps=1))
    assert len(ref) == len(got) > 5900
    assert np.allclose(ref, got, atol=1e-3)
