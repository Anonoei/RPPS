"""Batch kernels against the operators they are the batch form of, frame by frame (rpps.parallel)

Each kernel is run on a batch and every row compared with the operator run alone on that frame (the `LoopKernel` path, which
is the oracle): bits exactly, soft values to float32 rounding."""
from types import SimpleNamespace
import numpy as np
import pytest

import rpps as rp
from rpps.base.data import Mod
from rpps.base.kernel import Form, Batch, to_host
from rpps.parallel.kernel import LoopKernel, kernel_of
from rpps.profile.steps import DECODE, ENCODE, STAGE
from rpps.base.data import Digital
from support import fakedevice
from support.signals import named, noisy_symbols

def form(cat, dt, rep, cls=Mod):
    return Form(cls, cat, dt, rep, 1e6, 1, 0.0, 0.0)

def both(op, how, X, f, live=None, rtol=1e-5, atol=1e-4):
    """The operator's kernel and the loop on the same batch: results and which rows failed, compared"""
    step = SimpleNamespace(id="t", rx=[(how, op)], tx=[], controls={})
    mk = lambda: Batch(np.array(X), f, [0]*len(X), X.shape[1])
    ref = LoopKernel(step, live or {}).run(np, mk())
    k = kernel_of(step, live or {})
    assert not isinstance(k, LoopKernel), "the operator has no kernel"
    got = k.run(np, mk())
    assert list(got.bad) == list(ref.bad)
    assert got.x.shape == ref.x.shape and got.x.dtype == ref.x.dtype, (got.x.shape, ref.x.shape, got.x.dtype, ref.x.dtype)
    if got.x.dtype.kind in "fc":
        np.testing.assert_allclose(got.x, ref.x, rtol=rtol, atol=atol)
    else:
        np.testing.assert_array_equal(got.x, ref.x)
    assert tuple(got.form[:4]) == tuple(ref.form[:4])
    for xp in DEVICES:
        dev = on_device(k, mk(), xp)
        assert list(dev.bad) == list(got.bad) and dev.x.shape == got.x.shape and dev.x.dtype == got.x.dtype, xp.__name__
        if got.x.dtype.kind in "fc":
            np.testing.assert_allclose(dev.x, got.x, rtol=rtol, atol=atol, err_msg=xp.__name__)
        else:
            np.testing.assert_array_equal(dev.x, got.x, err_msg=xp.__name__)
    return k, got, ref

def _devices():
    """The pretend device always (tests/unit/fakedevice.py), and a real GPU when there is one (rpps._accel.device(): CuPy on
    CUDA or ROCm; RPPS_DEVICE=none leaves it out)"""
    from rpps import _accel
    xp, backend = _accel.device()
    return [fakedevice] + ([xp] if backend else [])

DEVICES = _devices()

def on_device(k, b, xp=fakedevice):
    """A kernel on a device, as a plan runs it there: uploaded, run (a host kernel between explicit transfers),
    downloaded once. On the pretend device anything implicit raises"""
    from rpps.base.kernel import ToDevice, ToHost
    b = ToDevice().run(xp, b)
    if k.where == "host":
        b = ToDevice().run(xp, k.run(np, ToHost().run(xp, b)))
    else:
        b = k.run(xp, b)
    b.x = to_host(b.x)
    assert type(b.x) is np.ndarray and type(b.bad) is np.ndarray
    return b

@pytest.mark.parametrize("name,noise", [("QPSK", 0.2), ("8PSK", 0.15), ("16QAM", 0.1), ("psk/qpsk dvb-s2", 0.3), ("psk/8psk dvb-s2", 0.1)])
@pytest.mark.parametrize("rows", [1, 5])
def test_demap(name, noise, rows):
    c = named(name)
    X = noisy_symbols(c, 700, rows, noise, 3)
    both(c, DECODE, X, form(rp.Category.SYMBOL, rp.dtype.SYMBOLS, rp.Rep.SYMBOLS))

def test_demap_max_log_and_a_controlled_constellation():
    c = rp.symbol.load("psk/qpsk", "dvb-s2")
    c.maxlog = True
    X = noisy_symbols(rp.symbol.load("psk/8psk", "dvb-s2"), 300, 3, 0.1, 4)
    k, got, _ = both(c, DECODE, X, form(rp.Category.SYMBOL, rp.dtype.SYMBOLS, rp.Rep.SYMBOLS), live={"name": "psk/8psk", "mapping": "dvb-s2"})
    assert got.x.shape == (3, 900) and k.describe(300)["out"] == 900

def test_demap_rows_do_not_see_each_other():
    """The noise estimate is per frame: a quiet frame next to a noisy one comes out as it does alone"""
    c = rp.symbol.load("QPSK")
    quiet, loud = noisy_symbols(c, 500, 1, 0.05, 1), noisy_symbols(c, 500, 1, 0.5, 2)
    f = form(rp.Category.SYMBOL, rp.dtype.SYMBOLS, rp.Rep.SYMBOLS)
    _, both_, _ = both(c, DECODE, np.vstack([quiet, loud]), f)
    _, alone, _ = both(c, DECODE, quiet, f)
    np.testing.assert_allclose(both_.x[0], alone.x[0], rtol=1e-5, atol=1e-4)

def test_demap_chunks_give_the_same():
    from rpps.symbol.kernels import Demap
    c = rp.symbol.load("8PSK")
    X = noisy_symbols(c, 400, 6, 0.1, 5)
    f = form(rp.Category.SYMBOL, rp.dtype.SYMBOLS, rp.Rep.SYMBOLS)
    k = kernel_of(SimpleNamespace(id="t", rx=[(DECODE, c)], tx=[]), {})
    whole = k.run(np, Batch(np.array(X), f, [0]*6, 400)).x
    Demap.CHUNK = 400*8*2 # two rows at a time
    try:
        parts = k.run(np, Batch(np.array(X), f, [0]*6, 400)).x
    finally:
        Demap.CHUNK = 1 << 22
    np.testing.assert_array_equal(whole, parts)

def coded_frames(code, blocks, rows, seed, noise=None, flips=0):
    """Rows of `blocks` codewords of `code`: hard bits with `flips` errors a codeword, or LLRs with Gaussian noise"""
    rng = np.random.default_rng(seed)
    k = code.e_num
    out = []
    for _ in range(rows):
        msg = rng.integers(0, 2, blocks*k).astype(bool)
        cw = np.asarray((rp.Data(msg, flush=True) * code).data).astype(bool)
        if noise is None:
            for w in range(blocks):
                for j in rng.choice(code.e_den, flips, replace=False):
                    cw[w*code.e_den + j] ^= True
            out.append(cw)
        else:
            out.append((3.0*(1 - 2.0*cw) + noise*rng.normal(size=len(cw))).astype(np.float32))
    return np.array(out)

HARD = form(rp.Category.CODE, rp.dtype.BITS, rp.Rep.CHANNEL, Digital)

SOFT = form(rp.Category.SYMBOL, rp.dtype.LLR, rp.Rep.CHANNEL, Mod)

@pytest.mark.parametrize("name", ["hamming/7_4", "golay/23_12", "bch/dvbs2_1_2"])
def test_block_code_hard(name):
    code = rp.code.load(f"block/{name}")
    X = coded_frames(code, 3, 4, 1, flips=1)
    k, got, _ = both(code, DECODE, X, HARD)
    assert got.x.shape[0] == 4

@pytest.mark.parametrize("name", ["ldpc/gallager_96_48", "ldpc/3_6"])
def test_block_code_soft(name):
    code = rp.code.load(f"block/{name}")
    X = coded_frames(code, 4, 5, 2, noise=0.8)
    both(code, DECODE, X, SOFT)

def test_ldpc_dvbs2_frame_batch():
    code = rp.code.load("block/ldpc/dvbs2_64800_1_2")
    X = coded_frames(code, 1, 3, 3, noise=1.0)
    both(code, DECODE, X, SOFT)

GPUS = [xp for xp in DEVICES if xp is not fakedevice]

def test_which_codes_have_device_kernels():
    """A layered LDPC code and a binary BCH code find their errors on a device; a permuting interleaver gathers; a code a
    frame switches off passes; other block codes decode on the host"""
    from rpps.code.kernels import BchDecode, BlockDecode, Identity, LdpcDecode, Permute
    k = lambda op, live=None: kernel_of(SimpleNamespace(id="t", rx=[(DECODE, op)], tx=[]), live or {})
    assert type(k(rp.code.load("block/ldpc/dvbs2_64800_1_2"))) is LdpcDecode
    assert type(k(rp.code.load("block/bch/dvbs2_1_2"))) is BchDecode
    assert type(k(rp.code.load("block/hamming/7_4"))) is BlockDecode
    assert type(k(rp.code.load("interleave/dvbs2_8psk"))) is Permute
    assert type(k(rp.code.load("interleave/dvbs2_8psk"), {"name": None})) is Identity

@pytest.mark.parametrize("flips", [0, 3, 12, 13])
@pytest.mark.parametrize("name", ["bch/dvbs2_1_2", "bch/dvbs2_short_1_2"])
def test_bch_on_every_device(name, flips):
    """Up to t errors a codeword corrected, more reported (code_failed), as the host decoder, on each device"""
    code = rp.code.load(f"block/{name}")
    X = coded_frames(code, 2, 3, 11 + flips, flips=flips)
    k, got, _ = both(code, DECODE, X, HARD)
    if flips <= 12:
        assert not got.code_failed.any()
    for xp in DEVICES:
        dev = on_device(k, Batch(np.array(X), HARD, [0]*3, X.shape[1]), xp)
        np.testing.assert_array_equal(dev.code_failed, got.code_failed)

def _bbframes(rows, kbch, dfl, seed, bad_crc=()):
    h = rp.link.load("header/dvbs2_bb")
    rng = np.random.default_rng(seed)
    out = []
    for i in range(rows):
        head = h.build({**h.values, "dfl": dfl, "syncd": int(rng.integers(0, 1 << 16))})
        if i in bad_crc:
            head[-1] ^= True
        out.append(np.concatenate((head, rng.integers(0, 2, kbch - len(head)).astype(bool))))
    return h, np.array(out)

BITS_DATA = form(rp.Category.CODE, rp.dtype.BITS, rp.Rep.DATA, Digital)

def test_header_kernel_is_the_operator():
    """The BBHEADER read in a batch: the payload `dfl` announces, as the operator one frame at a time"""
    from rpps.link.kernels import HeaderDecode
    h, X = _bbframes(4, 32208, 32128, 1)
    k, got, _ = both(h, DECODE, X, BITS_DATA, live={"length": 32208})
    assert type(k) is HeaderDecode and got.x.shape == (4, 32128)
    np.testing.assert_array_equal(got.x, X[:, 80:80 + 32128])

def test_header_kernel_drops_a_failed_crc_and_reads_nothing_of_a_short_row():
    h, X = _bbframes(3, 32208, 1000, 2, bad_crc=(1,))
    k = kernel_of(SimpleNamespace(id="t", rx=[(DECODE, h)], tx=[]), {"length": 32208})
    for xp in [np] + DEVICES:
        b = Batch(xp.asarray(X), BITS_DATA, [0]*3, X.shape[1])
        b = k.run(xp, b)
        assert list(b.bad) == [False, True, False] and "CRC" in b.why[1]
        np.testing.assert_array_equal(to_host(b.x)[[0, 2]], X[[0, 2], 80:1080])
    short = k.run(np, Batch(X[:, :40], BITS_DATA, [0]*3, 40))
    assert short.x.shape == (3, 0) and not short.bad.any()

def test_interleaver_switched_off_passes():
    il = rp.code.load("interleave/dvbs2_8psk")
    X = np.random.default_rng(3).normal(size=(3, 64800)).astype(np.float32)
    both(il, DECODE, X, SOFT, live={"name": None})

def test_ldpc_on_the_pretend_device_is_the_host_decoder():
    """Without kernels of its own (fakedevice), LdpcDecode moves the batch explicitly and decodes on the host"""
    code = rp.code.load("block/ldpc/dvbs2_64800_1_2")
    X = coded_frames(code, 1, 2, 7, noise=1.4)
    k = kernel_of(SimpleNamespace(id="t", rx=[(DECODE, code)], tx=[]), {})
    ref = k.run(np, Batch(np.array(X), SOFT, [0, 0], X.shape[1]))
    dev = on_device(k, Batch(np.array(X), SOFT, [0, 0], X.shape[1]), fakedevice)
    np.testing.assert_array_equal(dev.x, ref.x)
    np.testing.assert_array_equal(dev.code_failed, ref.code_failed)

@pytest.mark.skipif(not GPUS, reason="no GPU (rpps._accel.device())")
@pytest.mark.parametrize("name,noise", [("dvbs2_64800_1_2", 1.0), ("dvbs2_64800_3_4", 0.7), ("dvbs2_16200_8_9", 0.45)])
def test_device_ldpc_equals_the_host_where_it_converges(name, noise):
    """float32 on the device, float64 on the host: the same codewords and verdicts above threshold, several per row"""
    code = rp.code.load(f"block/ldpc/{name}")
    X = coded_frames(code, 2, 6, 8, noise=noise)
    k = kernel_of(SimpleNamespace(id="t", rx=[(DECODE, code)], tx=[]), {})
    ref = k.run(np, Batch(np.array(X), SOFT, [0]*6, X.shape[1]))
    dev = on_device(k, Batch(np.array(X), SOFT, [0]*6, X.shape[1]), GPUS[0])
    assert not ref.code_failed.any()
    np.testing.assert_array_equal(dev.x, ref.x)
    np.testing.assert_array_equal(dev.code_failed, ref.code_failed)

@pytest.mark.skipif(not GPUS, reason="no GPU (rpps._accel.device())")
def test_device_ldpc_near_threshold_fails_no_more_frames_than_the_host():
    """At the waterfall float32 and float64 part on a few ties: frame failures within a margin measured over seeds"""
    code = rp.code.load("block/ldpc/dvbs2_64800_1_2")
    k = kernel_of(SimpleNamespace(id="t", rx=[(DECODE, code)], tx=[]), {})
    host = dev = 0
    for seed in range(3):
        X = coded_frames(code, 1, 8, 20 + seed, noise=2.75)
        host += int(k.run(np, Batch(np.array(X), SOFT, [0]*8, X.shape[1])).code_failed.sum())
        dev += int(on_device(k, Batch(np.array(X), SOFT, [0]*8, X.shape[1]), GPUS[0]).code_failed.sum())
    assert 0 < host < 24, host # (at the waterfall: some frames fail, not all)
    assert abs(dev - host) <= 3, (dev, host)

def test_block_interleaver():
    il = rp.code.load("interleave/dvbs2_8psk")
    rng = np.random.default_rng(6)
    X = rng.normal(size=(3, 2*il._block(False))).astype(np.float32)
    both(il, DECODE, X, SOFT)

def test_forney_interleaver_and_encoding_have_no_kernel():
    from rpps.parallel.kernel import kernel_of
    fy = rp.code.load("interleave/forney/12_17")
    c = rp.code.load("block/hamming/7_4")
    for op, how in ((fy, DECODE), (c, ENCODE)):
        step = SimpleNamespace(id="t", rx=[(how, op)], tx=[])
        assert isinstance(kernel_of(step, {}), LoopKernel)

def test_a_frame_that_is_not_whole_blocks_reports_the_rest():
    code = rp.code.load("block/hamming/7_4")
    X = np.concatenate((coded_frames(code, 2, 3, 7, flips=1), np.zeros((3, 3), bool)), axis=1)
    k, got, ref = both(code, DECODE, X, HARD)
    assert got.notes == ref.notes and got.notes[0][0]["items"] == 3

def test_thread_safe_ldpc_state_stays_per_call():
    """Two decodes of one code on different threads: each result is its own (the code keeps nothing a decode reads back)"""
    from concurrent.futures import ThreadPoolExecutor
    code = rp.code.load("block/ldpc/gallager_96_48")
    A, B = coded_frames(code, 8, 1, 1, noise=0.7)[0], coded_frames(code, 8, 1, 2, noise=0.7)[0]
    want = [code.decoder.decode(A).copy(), code.decoder.decode(B).copy()]
    with ThreadPoolExecutor(4) as pool:
        got = list(pool.map(lambda x: code.decoder.decode(x).copy(), [A, B]*8))
    assert all(np.array_equal(g, want[i % 2]) for i, g in enumerate(got))

def test_demap_exact_where_the_sums_underflow():
    """Near noiseless symbols and outliers far from every point: the exponentials underflow, the long way gives the LLRs"""
    c = rp.symbol.load("8PSK")
    rng = np.random.default_rng(1)
    s = c.points[rng.integers(0, 8, (2, 3000))]
    s = (s + 1e-5*(rng.normal(size=s.shape) + 1j*rng.normal(size=s.shape))).astype(np.complex64)
    s[0, ::97] *= 3
    _, got, _ = both(c, DECODE, s, form(rp.Category.SYMBOL, rp.dtype.SYMBOLS, rp.Rep.SYMBOLS), rtol=1e-6, atol=1e-3)
    assert np.isfinite(got.x).all() and np.abs(got.x).max() > 1e5

def reference_llr(c, syms):
    """The definition, plainly: EM noise estimate (8 iterations), then the log of the sums over the points of each bit value"""
    from rpps.math.prob import logsumexp, mixture_variance
    y = np.asarray(syms, np.complex128)
    d2 = np.abs(y[:, None] - c.points[None, :])**2
    nv = max(2*mixture_variance(d2, 8, 1e-6), 2e-6)
    logp = -d2/nv
    lab = c.labels().astype(bool)
    return np.stack([logsumexp(logp[:, ~lab[:, j]]) - logsumexp(logp[:, lab[:, j]]) for j in range(c.bits_per_symbol)], axis=1).reshape(-1)

@pytest.mark.parametrize("name,noise", [("QPSK", 0.3), ("8PSK", 0.12), ("16QAM", 0.08), ("psk/8psk dvb-s2", 0.1)])
def test_llr_is_the_definition(name, noise):
    """The shared implementation against the plain log-sum-exp, which it replaced"""
    c = named(name)
    x = noisy_symbols(c, 600, 1, noise, 9)[0]
    np.testing.assert_allclose(c.llr(x), reference_llr(c, x), rtol=1e-5, atol=1e-4)

# ---- structure among symbols and bits: scrambling, pilots, the carrier line

SYM = form(rp.Category.SYMBOL, rp.dtype.SYMBOLS, rp.Rep.SYMBOLS)

def rand_symbols(rows, n, seed, turn=0.0):
    rng = np.random.default_rng(seed)
    p = np.exp(1j*(np.pi/4 + np.pi/2*rng.integers(0, 4, (rows, n))))
    return (p*np.exp(1j*turn) + 0.05*(rng.normal(size=(rows, n)) + 1j*rng.normal(size=(rows, n)))).astype(np.complex64)

@pytest.mark.parametrize("how", [DECODE, ENCODE])
@pytest.mark.parametrize("n,length", [(2500, None), (2500, 1000), (777, 777)])
def test_phase_scramble(how, n, length):
    ps = rp.symbol.PhaseScramble(rp.sequence.load("dvbs2_pl"), [131072, 0])
    both(ps, how, rand_symbols(4, n, 1), SYM, live={"length": length} if length else {})

@pytest.mark.parametrize("pilots", [1, 0])
@pytest.mark.parametrize("delta", [0, 40, -50])
def test_pilot_removal(pilots, delta):
    pl = rp.symbol.Pilots(slots=20, pilots=bool(pilots))
    total, _ = rp.symbol.frame.pilot_layout(20, pilots)
    both(pl, DECODE, rand_symbols(3, total + delta, 2), SYM, live={"slots": 20, "pilots": pilots})

@pytest.mark.parametrize("pilots", [1, 0])
@pytest.mark.parametrize("delta", [0, 40, -50])
@pytest.mark.parametrize("lead", [None, -77.5])
def test_pilot_line(pilots, delta, lead):
    """A frame whose carrier turns along it: the line through its pilot blocks (and the lead) is taken off, row by row"""
    total, _ = rp.symbol.frame.pilot_layout(20, pilots)
    n = total + delta
    ramp = 0.3 + 0.002*np.arange(n)
    X = (rand_symbols(3, n, 3)*np.exp(1j*ramp)[None, :]).astype(np.complex64)
    X[:, :0] = X[:, :0]
    pp = rp.sync.PilotPhase(slots=20, pilots=bool(pilots), lead=lead)
    both(pp, STAGE, X, SYM, live={"slots": 20, "pilots": pilots})

@pytest.mark.parametrize("how", [DECODE, ENCODE])
@pytest.mark.parametrize("n,length", [(800, 800), (2000, 800)])
@pytest.mark.parametrize("soft", [False, True])
def test_additive_scrambler_restarted_every_frame(how, n, length, soft):
    sc = rp.code.scram.load("additive/dvbs2_bb")
    rng = np.random.default_rng(5)
    bits = rng.integers(0, 2, (3, n)).astype(bool)
    X = (3.0*(1 - 2.0*bits) + rng.normal(size=bits.shape)).astype(np.float32) if soft else bits
    f = SOFT if soft else form(rp.Category.CODE, rp.dtype.BITS, rp.Rep.DATA, Digital)
    both(sc, how, X, f, live={"length": length})

def test_scramblers_without_a_frame_length_or_with_memory_have_no_kernel():
    for name, kw in (("additive/dvbs2_bb", {}), ("additive/lte_pbch", {"pci": 1}), ("feedthrough/v35", {})):
        sc = rp.code.scram.load(name, **kw)
        step = SimpleNamespace(id="t", rx=[(DECODE, sc)], tx=[])
        assert isinstance(kernel_of(step, {}), LoopKernel), name
