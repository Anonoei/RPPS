"""Spreading codes (against their standards) and the Despread stage"""
import numpy as np
import pytest

import rpps as rp
from rpps import _accel

# IS-GPS-200 Table 3-Ia: the first 10 chips of PRN 1..10 in octal
GPS_FIRST10 = [0o1440, 0o1620, 0o1710, 0o1744, 0o1133, 0o1455, 0o1131, 0o1454, 0o1626, 0o1504]

def test_gps_first_chips():
    for prn, want in enumerate(GPS_FIRST10, 1):
        b = rp.sequence.load("gps_ca", prn=prn).bits()[0][:10]
        assert int("".join(map(str, b)), 2) == want, prn

def test_gps_autocorrelation():
    for prn in (1, 17, 32):
        c = rp.sequence.load("gps_ca", prn=prn)
        ac = c.autocorrelation()
        assert c.length == 1023 and ac[0] == 1023
        assert set(ac[1:]) <= {-65, -1, 63}

def test_gps_cross_correlation_bounded():
    a, b = rp.sequence.load("gps_ca", prn=3), rp.sequence.load("gps_ca", prn=21)
    both = rp.sequence.Code("pair", np.vstack((a.chips, b.chips)))
    assert set(both.crosscorrelation(0, 1)) <= {-65, -1, 63}

def test_gps_prn_out_of_table():
    with pytest.raises(rp.errors.ConfigError):
        rp.sequence.load("gps_ca", prn=33)

def test_barker11():
    c = rp.sequence.load("wifi_dsss") # alias
    assert c.length == 11 and c.rate == 11e6
    assert c.autocorrelation()[0] == 11 and np.abs(c.autocorrelation()[1:]).max() == 1
    ac = np.correlate(c[0], c[0], "full") # aperiodic sidelobes of a Barker code are at most 1
    assert np.abs(ac[ac.size//2 + 1:]).max() <= 1

def test_802154_structure():
    c = rp.sequence.load("ieee802154_2450")
    b = c.bits(one=1)
    assert c.count == 16 and c.length == 32
    for k in range(1, 8): # symbols 1..7: symbol 0 rotated right by 4 chips each
        assert np.array_equal(b[k], np.roll(b[0], 4*k))
    for k in range(8): # symbols 8..15: the others with every odd chip inverted
        want = b[k].copy()
        want[1::2] ^= 1
        assert np.array_equal(b[k + 8], want)
    off = c.chips @ c.chips.T
    assert np.all(np.diag(off) == 32) and np.abs(off - 32*np.eye(16)).max() < 32

def test_spread_despread_roundtrip():
    rng = np.random.default_rng(0)
    code = rp.sequence.load("barker11")
    sym = rng.choice([-1, 1], 50) + 0j
    assert np.allclose(code.despread(code.spread(sym)), sym)

def run_stage(x, blocks, native, **kw):
    saved = _accel.native
    _accel.native = saved if native else None
    try:
        stage = rp.symbol.Despread(rp.sequence.load("barker11"), **kw)
        stream, outs, i = None, [], 0
        for b in blocks:
            d = rp.Data(x[i:i + b], stream=stream, Fs=44e6, sps=4)
            stream = d.stream
            outs.append((d + stage).data)
            i += b
        return np.concatenate(outs)
    finally:
        _accel.native = saved

@pytest.fixture
def chips44():
    rng = np.random.default_rng(1)
    code = rp.sequence.load("barker11")
    sym = np.exp(1j*np.pi/2*rng.integers(0, 4, 60))
    x = np.repeat(code.spread(sym), 4) + 0.1*(rng.standard_normal(60*44) + 1j*rng.standard_normal(60*44))
    return sym, np.concatenate((np.zeros(7), x)).astype(np.complex64)

def test_despread_stage_recovers_symbols(chips44):
    sym, x = chips44
    out = run_stage(x, [len(x)], native=False, offset=7)
    assert len(out) == len(sym)
    assert np.abs(out - sym).max() < 0.25

@pytest.mark.parametrize("native", [False, pytest.param(True, marks=pytest.mark.skipif(not _accel.available(), reason="rpps._native is not built"))])
def test_despread_block_cuts(chips44, native):
    _, x = chips44
    ref = run_stage(x, [len(x)], native=False, offset=7)
    n = len(x)
    for cuts in ([n//3, n//3, n - 2*(n//3)], [1]*40 + [n - 40], [44, 43, n - 87]):
        got = run_stage(x, cuts, native=native, offset=7)
        assert len(got) == len(ref)
        assert np.allclose(got, ref, atol=1e-6)

@pytest.mark.skipif(not _accel.available(), reason="rpps._native is not built")
def test_native_matches_python_set():
    rng = np.random.default_rng(2)
    code = rp.sequence.load("ieee802154_2450")
    x = (rng.standard_normal(32*3*20) + 1j*rng.standard_normal(32*3*20)).astype(np.complex64)
    saved = _accel.native
    try:
        nat = rp.symbol.correlate(x, code, 3)
        _accel.native = None
        py = rp.symbol.correlate(x, code, 3)
    finally:
        _accel.native = saved
    assert nat.shape == (20, 16) and np.allclose(nat, py, atol=1e-5)

def test_set_detection_802154():
    rng = np.random.default_rng(3)
    code = rp.sequence.load("ieee802154_2450")
    syms = rng.integers(0, 16, 40)
    chips = code.spread(np.ones(40), index=syms) + 0.5*rng.standard_normal(40*32)
    corr = rp.symbol.correlate(chips.astype(np.complex64), code, 1)
    assert np.array_equal(np.abs(corr).argmax(axis=1), syms)

# ---- the Spread operator, in the chains of the standards

from rpps.symbol import Differential

def hard(d):
    return np.asarray(d.data) < 0 if d.DT == rp.dtype.LLR else np.asarray(d.data).astype(bool)

def samples(like, x, ctx=None):
    """`x` as chips arriving at a receiver, in the stream of `like` or a fresh one"""
    return like.derive(x, rp.Category.SYMBOL, rp.dtype.SYMBOLS, rp.base.Mod, {} if ctx is None else ctx)

def wifi(rate):
    """802.11b DSSS: DBPSK (1 Mb/s) or DQPSK (2 Mb/s), differential, spread by Barker-11"""
    const = rp.symbol.load("bpsk" if rate == 1 else "qpsk_axes")
    return const, Differential(), rp.symbol.Spread(rp.sequence.load("barker11"))

@pytest.mark.parametrize("rate,chips_per_bit", [(1, 11), (2, 5.5)])
def test_wifi_dsss_roundtrip(rate, chips_per_bit):
    rng = np.random.default_rng(4)
    msg = rng.integers(0, 256, 24).astype(np.uint8)
    const, diff, spread = wifi(rate)
    chips = rp.Data(msg) * const * diff * spread
    assert len(chips.data) == pytest.approx(24*8*chips_per_bit)
    back = chips / spread / diff / const
    assert np.array_equal(hard(back), np.unpackbits(msg).astype(bool))

def test_gps_ca_bpsk_roundtrip():
    rng = np.random.default_rng(5)
    msg = rng.integers(0, 256, 4).astype(np.uint8)
    spread = rp.symbol.Spread(rp.sequence.load("gps_ca", prn=9))
    bpsk = rp.symbol.load("bpsk")
    chips = rp.Data(msg) * bpsk * spread
    assert len(chips.data) == 32*1023 and chips.Fs == pytest.approx(rp.Data(msg).Fs*1023)
    assert np.array_equal(hard(chips / spread / bpsk), np.unpackbits(msg).astype(bool))

def test_spread_sets_rates_and_sps():
    d = rp.Data(np.array([1, -1, 1, 1], dtype=np.complex64), Fs=1e6, sps=1)
    d = samples(d, d.data)
    code = rp.sequence.load("barker11")
    chips = d * rp.symbol.Spread(code)
    assert chips.Fs == pytest.approx(11e6) and chips.sps == 1
    assert (chips / rp.symbol.Spread(code)).Fs == pytest.approx(1e6)

@pytest.mark.parametrize("pair", [False, True])
def test_zigbee_set_roundtrip(pair):
    rng = np.random.default_rng(6)
    msg = rng.integers(0, 256, 24).astype(np.uint8)
    spread = rp.symbol.Spread(rp.sequence.load("ieee802154_2450"), pair=pair)
    chips = rp.Data(msg) * spread
    assert len(chips.data) == 24*2*(16 if pair else 32) # two 4-bit symbols per byte
    assert (chips.data.dtype.kind == "c")
    assert np.array_equal(hard(chips / spread), np.unpackbits(msg).astype(bool))

def test_zigbee_pair_puts_even_chips_on_i_and_odd_on_q():
    code = rp.sequence.load("ieee802154_2450")
    chips = (rp.Data(np.array([0b0000_0101], dtype=np.uint8)) * rp.symbol.Spread(code, pair=True)).data
    # bits come most significant first: 0 0 0 0, then 0 1 0 1; the first bit of a symbol is its least significant
    want = []
    for idx in (0, 0b1010): # the first bit is the least significant: bits 0,1,0,1 -> 0b1010
        c = code[idx]
        want += list(c[0::2] + 1j*c[1::2])
    assert np.allclose(chips, want)

def test_dsss_noise_and_rotation():
    """DQPSK through the 11-chip code: noise 0.5 per chip and a constant carrier phase cost almost nothing"""
    rng = np.random.default_rng(5)
    msg = rng.integers(0, 256, 40).astype(np.uint8)
    const, diff, spread = wifi(2)
    tx = rp.Data(msg) * const * diff * spread
    rx = tx.data*np.exp(1j*0.7) + 0.5*(rng.standard_normal(len(tx.data)) + 1j*rng.standard_normal(len(tx.data)))
    got = hard(samples(tx, rx.astype(np.complex64), {}) / spread / diff / const)
    want = np.unpackbits(msg).astype(bool)
    assert (got[2:] != want[2:]).mean() < 0.02 # the first symbol has no reference to be a change from; a constant phase cancels after it

def test_wifi_dqpsk_phase_changes():
    """802.11b DQPSK: dibits 00 01 11 10 are phase changes of 0, 90, 180, 270 degrees"""
    const, diff, spread = wifi(2)
    bits = np.packbits(np.array([0,0, 0,1, 1,1, 1,0], dtype=np.uint8))
    sym = (rp.Data(bits) * const * diff).data
    ph = np.round(np.degrees(np.angle(sym[1:]*np.conj(sym[:-1]))) % 360)
    assert list(ph) == [90, 180, 270] and abs(np.degrees(np.angle(sym[0])) % 360) < 1e-3

def test_dsss_stream_blocks():
    rng = np.random.default_rng(6)
    msg = rng.integers(0, 256, 64).astype(np.uint8)
    const, diff, spread = wifi(1)
    tx = rp.Data(msg) * const * diff * spread
    chips = tx.data
    outs, i, ctx = [], 0, None
    for b in (100, 7, 1, 300, len(chips) - 408):
        r = samples(tx, chips[i:i + b], ctx) / spread / diff / const
        ctx = dict(r.ctx)
        outs.append(hard(r))
        i += b
    assert np.array_equal(np.concatenate(outs), np.unpackbits(msg).astype(bool))

# ---- acquisition

def gps_signal(rng, prn=7, sps=2, delay=357, dop=1300.0, snr_db=-10.0, periods=8):
    code = rp.sequence.load("gps_ca", prn=prn)
    Fs = 1.023e6*sps
    n = code.length*sps*periods
    data = np.repeat(rng.choice([-1, 1], periods), code.length*sps) # a bit flip now and then is in the way of coherent summing
    sig = np.tile(np.repeat(code[0], sps), periods)*data
    sig = np.roll(sig, delay)*np.exp(2j*np.pi*dop*np.arange(n)/Fs)
    sigma = 10**(-snr_db/20)/np.sqrt(2)
    return code, Fs, (sig + sigma*(rng.standard_normal(n) + 1j*rng.standard_normal(n))).astype(np.complex64)

def test_acquire_gps_code_phase_and_doppler():
    rng = np.random.default_rng(7)
    code, Fs, x = gps_signal(rng)
    r = rp.sync.acquire_code(x, code, 2, Fs=Fs, doppler=np.arange(-5000, 5001, 500))
    assert r.locked and r.offset == 357 and abs(r.doppler - 1300) <= 250

def test_acquire_wrong_prn_is_not_locked():
    rng = np.random.default_rng(8)
    _, Fs, x = gps_signal(rng, prn=7)
    other = rp.sequence.load("gps_ca", prn=12)
    r = rp.sync.acquire_code(x, other, 2, Fs=Fs, doppler=np.arange(-5000, 5001, 500))
    assert not r.locked

def test_acquire_then_despread():
    rng = np.random.default_rng(9)
    code = rp.sequence.load("barker11")
    sym = np.exp(1j*np.pi*rng.integers(0, 2, 80))
    x = np.repeat(code.spread(sym), 4)
    x = np.concatenate((0.05*rng.standard_normal(19), x)) + 0.3*(rng.standard_normal(len(x) + 19) + 1j*rng.standard_normal(len(x) + 19))
    r = rp.sync.acquire_code(x.astype(np.complex64), code, 4)
    assert r.locked and r.offset == 19
    out = (rp.Data(x.astype(np.complex64), Fs=44e6, sps=4) + rp.symbol.Despread(code, offset=r.offset)).data
    assert np.array_equal(np.sign(out.real), np.sign(sym.real[:len(out)]))
