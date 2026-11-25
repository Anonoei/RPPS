"""The decoder kernels (Viterbi, BCJR, LDPC, Reed-Solomon, BCH, polar) give the Python reference's output (off and on)"""
import numpy as np
import pytest

import rpps as rp
from rpps import _accel
from support.native import both, kernels
from support.signals import bpsk_llr

def noisy_llr(bits, rng, ebn0=2.0):
    s = 1.0 - 2.0*np.asarray(bits, dtype=np.float64)
    sigma = 10**(-ebn0/20)
    return 2*(s + sigma*rng.standard_normal(len(s)))/sigma**2

# ---------------------------------------------------------------- Viterbi (conv) and fsm
@pytest.mark.parametrize("name", ["k7_133_171/1_2", "k7_171_133/3_4", "k3_5_7/1_2"])
@pytest.mark.parametrize("soft", [True, False])
def test_viterbi_frames(name, soft, rng):
    code = rp.code.load(f"conv/{name}")
    msg = rng.integers(0, 2, 600).astype(bool)
    llr = noisy_llr(code.encoder.encode(msg), rng, ebn0=6.0)
    x = llr if soft else (llr < 0)
    ref, got = both(lambda: code.decoder.decode(x))
    assert np.array_equal(ref, got)
    n = min(len(got), len(msg)) - 10 # the end of a truncated code is unsettled
    assert np.mean(got[:n] != msg[:n]) < 0.02

def test_viterbi_stream_carries_state(rng):
    code = rp.code.load("conv/k7_133_171/1_2")
    llr = noisy_llr(code.encoder.encode(rng.integers(0, 2, 2000).astype(bool)), rng)
    def stream():
        st, out = {}, []
        for i in range(0, len(llr), 302):
            part = llr[i:i+302]
            out.append(code.decoder.decode(part, soft=True, state=st, final=i + 302 >= len(llr)))
        return np.concatenate(out)
    ref, got = both(stream)
    assert np.array_equal(ref, got)

def test_viterbi_tailbiting(rng):
    code = rp.code.Conv([0o133, 0o171, 0o165], 7, termination="tailbiting")
    msg = rng.integers(0, 2, 40).astype(bool)
    llr = noisy_llr(code.encoder.encode(msg), rng, ebn0=3.0)
    ref, got = both(lambda: code.decoder.decode(llr))
    assert np.array_equal(ref, got) and np.array_equal(got, msg)

@pytest.mark.parametrize("name", ["trellis/p25_1_2", "trellis/atsc_8vsb"])
def test_fsm_batches(name, rng):
    code = rp.code.load(f"block/{name}").decoder
    msg = rng.integers(0, 2, (12, 2*96)).astype(bool)
    enc = np.array([code.encode(m) for m in msg])
    if code.levels is not None: # amplitudes
        x = np.array([code.modulate(e) for e in enc]) + 0.6*rng.standard_normal((12, len(enc[0])//code.sym_bits))
    else:
        x = np.array([noisy_llr(e, rng, ebn0=3.0) for e in enc])
    ref, got = both(lambda: code.decode_batch(x, status=True))
    assert np.array_equal(ref[0], got[0])
    assert (ref[1] is None and got[1] is None) or np.array_equal(ref[1], got[1])

# ---------------------------------------------------------------- BCJR: conv, turbo, syndrome trellis
@pytest.mark.parametrize("maxlog", [False, True])
def test_bcjr_conv(maxlog, rng):
    code = rp.code.load("conv/k7_133_171/1_2", decoder="bcjr")
    code.decoder.maxlog = maxlog
    llr = noisy_llr(code.encoder.encode(rng.integers(0, 2, 300).astype(bool)), rng)
    ref, got = both(lambda: code.decoder.app(llr, soft=True))
    assert np.allclose(ref, got, rtol=1e-9, atol=1e-9)

@pytest.mark.parametrize("maxlog", [True, False])
def test_turbo(maxlog, rng):
    code = rp.code.load("block/turbo/lte", K=104)
    code.decoder.maxlog = maxlog
    msg = rng.integers(0, 2, 2*104).astype(bool)
    llr = noisy_llr(code.encoder.encode(msg), rng, ebn0=0.5)
    ref, got = both(lambda: (code.decoder.decode(llr), code.decoder.app(llr)))
    assert np.array_equal(ref[0], got[0])
    assert np.allclose(ref[1], got[1], rtol=1e-6, atol=1e-6)
    assert np.array_equal(got[0], got[1] < 0) # decode's bits are app's signs

def test_syndrome_trellis(rng):
    code = rp.code.load("block/hamming/15_11", decoder="trellis")
    words = code.encoder.encode(rng.integers(0, 2, 11*20).astype(bool))
    llr = noisy_llr(words, rng, ebn0=4.0)
    ref, got = both(lambda: code.decoder.decode(llr, soft=True))
    assert np.array_equal(ref, got)

# ---------------------------------------------------------------- LDPC
def test_ldpc_one_codeword_at_a_time(rng):
    code = rp.code.load("block/ldpc/gallager_96_48")
    k = code.decoder.num # 50: the construction's H has dependent rows
    for _ in range(10):
        llr = noisy_llr(code.encoder.encode(rng.integers(0, 2, k).astype(bool)), rng, ebn0=2.5)
        ref, got = both(lambda: code.decoder.decode(llr, status=True))
        assert np.array_equal(ref[0], got[0]) and np.array_equal(ref[1], got[1])

def test_ldpc_batch_is_each_codeword_on_its_own(rng):
    """The kernel stops each codeword when it converges: the numpy decoder run one codeword at a time"""
    code = rp.code.load("block/ldpc/gallager_96_48")
    k, n = code.decoder.num, code.decoder.den
    llr = noisy_llr(code.encoder.encode(rng.integers(0, 2, k*40).astype(bool)), rng, ebn0=2.5)
    def each():
        out, conv = [], []
        for w in llr.reshape(40, n):
            o, e = code.decoder.decode(w, status=True)
            out.append(o)
            conv.append(e[0] >= 0)
        return np.array(out), np.array(conv)
    with kernels(False):
        ref = each()
    o, e = code.decoder.decode(llr, status=True)
    got = o.reshape(40, k), e >= 0
    assert np.array_equal(ref[0], got[0]) and np.array_equal(ref[1], got[1]) and got[1].sum() > 30

# ---------------------------------------------------------------- Reed-Solomon, BCH
def corrupt_symbols(code, msg_syms, rng, n_err, m=8):
    cw = code.encoder.encode(msg_syms, symbols=True).reshape(-1, code.decoder.den).copy()
    for w in cw:
        pos = rng.choice(len(w), n_err, replace=False)
        w[pos] ^= rng.integers(1, 1 << m, n_err)
    return cw

@pytest.mark.parametrize("name, n_err", [("rs/204_188", 8), ("rs/204_188", 9), ("rs/p25_24_12", 6), ("rs/p25p2_soemi", 2)])
def test_rs_errors(name, n_err, rng):
    code = rp.code.load(f"block/{name}")
    k, m = code.decoder.num, code.decoder.m
    cw = corrupt_symbols(code, rng.integers(0, 1 << m, k*30), rng, n_err, m)
    ref, got = both(lambda: code.decoder.decode(cw.reshape(-1), symbols=True, status=True))
    assert np.array_equal(ref[0], got[0]) and np.array_equal(ref[1], got[1])

def test_rs_erasures_and_soft_retries(rng):
    code = rp.code.load("block/rs/204_188")
    msg = rng.integers(0, 256, 188*20)
    cw = corrupt_symbols(code, msg, rng, 0)
    erased = np.zeros(cw.shape, dtype=bool)
    for w, e in zip(cw, erased):
        pos = rng.choice(204, 12, replace=False)
        w[pos] ^= rng.integers(1, 256, 12)
        e[pos[:8]] = True # 8 of the 12 known: 8 erasures + 4 errors <= 16
    ref, got = both(lambda: code.decoder.decode(cw.reshape(-1), symbols=True, erasures=erased, status=True))
    assert np.array_equal(ref[0], got[0]) and np.array_equal(ref[1], got[1]) and (got[1] >= 0).all()
    # soft: 10 errors (beyond t = 8), the corrupted bytes least reliable
    cw = corrupt_symbols(code, msg, rng, 10)
    bits = code.decoder.to_bits(cw.reshape(-1)).reshape(20, 204, 8)
    llr = np.where(bits, -4.0, 4.0) + 0.1*rng.standard_normal(bits.shape)
    clean = code.decoder.to_bits(code.encoder.encode(msg, symbols=True)).reshape(20, 204, 8)
    llr[bits != clean] = np.where(bits[bits != clean], -0.3, 0.3)
    ref, got = both(lambda: code.decoder.decode(llr.reshape(-1), status=True))
    assert np.array_equal(ref[0], got[0]) and np.array_equal(ref[1], got[1])

def test_bch(rng):
    code = rp.code.load("block/bch/dvbs2_1_4")
    k, n = code.decoder.num, code.decoder.den
    cw = code.encoder.encode(rng.integers(0, 2, k*3).astype(bool)).reshape(3, n).copy()
    for w, ne in zip(cw, (5, 12, 14)):
        w[rng.choice(n, ne, replace=False)] ^= True
    ref, got = both(lambda: code.decoder.decode(cw.reshape(-1), status=True))
    assert np.array_equal(ref[0], got[0]) and np.array_equal(ref[1], got[1])
    assert list(got[1][:2]) == [5, 12]

# --- Polar list decoding: polar_scl against the numpy reference (rpps/code/types/polar.py _scl)

@pytest.mark.parametrize("N,size", [(8, 1), (32, 4), (128, 8), (512, 8), (1024, 16)])
def test_polar_scl_matches_python(N, size, rng):
    """The same paths in the same order and the same metrics: candidates ranked stably, 0s before 1s on a tie"""
    import sys
    pol = sys.modules["rpps.code.types.polar"]
    frozen = np.zeros(N, bool)
    frozen[pol.pw_order(N)[:N - max(1, N//3)]] = True
    llr = rng.normal(1.0, 1.5, (4, N))*rng.choice([-1, 1], (4, N))
    llr[0, :4] = 0.0 # (ties)
    u1, m1 = _accel.native.polar_scl(llr, frozen, size)
    u2, m2 = pol._scl(llr, frozen, size)
    assert np.array_equal(u1, u2) and np.allclose(m1, m2, rtol=1e-12, atol=1e-9)

@pytest.mark.parametrize("level", ["balanced", "max"])
def test_ldpc_native_matches_numpy(level):
    dec = rp.code.load("block/ldpc/ieee80211n_648_1_2").decoder
    rng = np.random.default_rng(6)
    m = rng.integers(0, 2, 16*dec.num).astype(bool)
    llr = bpsk_llr(dec.encode(m), 1.5, dec.rate, rng).reshape(16, dec.N)
    with rp.effort(level):
        nat = dec.layered(llr)
        with kernels(False):
            ref = dec.layered(llr)
    np.testing.assert_array_equal(nat[1], ref[1])
    np.testing.assert_array_equal(nat[0][nat[1]], ref[0][ref[1]])
