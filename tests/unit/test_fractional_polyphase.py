"""FractionalPolyphase: any ratio, the stream does not depend on the block size, the fused shift is the shift then the resampler"""
import numpy as np
import pytest

import rpps as rp
from rpps import _accel
from rpps.signal.sample import FractionalPolyphase, fractional_resample, resample_poly
from rpps.signal.sample.fractional import design_bank, read_python

Fs = 20e6

@pytest.fixture(params=["python", "native"])
def impl(request, monkeypatch):
    if request.param == "python":
        monkeypatch.setattr(_accel, "native", None)
    elif _accel.native is None:
        pytest.skip("rpps._native not built")
    return request.param

def noise(n, dtype=np.complex64, seed=0):
    r = np.random.default_rng(seed)
    x = r.standard_normal(n) + (1j*r.standard_normal(n) if np.dtype(dtype).kind == "c" else 0)
    return x.astype(dtype)

def run(make, x, blocks, rates=None, **kw):
    """x cut into blocks of one stream through a fresh stage: (output, the last block's rates)"""
    stage = make()
    stream, outs, i, d = None, [], 0, None
    for b in blocks:
        d = rp.Data(x[i:i + b], stream=stream, **(rates or dict(Fs=Fs, sps=2, cf=604e6)))
        stream = d.stream
        d = d + stage
        outs.append(np.asarray(d.data))
        i += b
    return np.concatenate(outs), d

def cuts(n):
    return [[n], [5000]*(n//5000), [777]*(n//777) + [n % 777], [1]*40 + [n - 40], [0, n]]

RATIO = 10.7622/20

@pytest.mark.parametrize("dtype", [np.complex64, np.float32])
@pytest.mark.parametrize("shift", [0.0, -0.5e6])
def test_block_size_does_not_change_the_stream(impl, dtype, shift):
    x = noise(30000, dtype)
    make = lambda: FractionalPolyphase(RATIO, cutoff=3.7e6, shift=shift)
    ref, _ = run(make, x, [len(x)])
    for sp in cuts(len(x))[1:]:
        y, _ = run(make, x, sp)
        assert len(y) == len(ref)
        np.testing.assert_allclose(y, ref, atol=2e-5)

def test_rates_and_dtypes(impl):
    x = noise(8000)
    y, d = run(lambda: FractionalPolyphase(RATIO, shift=-0.5e6), x, [8000])
    assert y.dtype == np.complex64
    assert d.Fs == pytest.approx(Fs*RATIO) and d.sps == pytest.approx(2*RATIO) and d.cf == pytest.approx(604e6 - 0.5e6)
    assert abs(len(y)/len(x) - RATIO) < 0.01
    yr, _ = run(lambda: FractionalPolyphase(0.7), noise(5000, np.float32), [5000], dict(Fs=1e3, sps=1))
    assert yr.dtype == np.float32

def test_dc_gain_is_one_everywhere(impl):
    for ratio in (0.37, 0.5379, 1.0, 1.9):
        y, _ = run(lambda: FractionalPolyphase(ratio), np.ones(4000, np.complex64), [4000], dict(Fs=1e3, sps=1))
        np.testing.assert_allclose(y[40:-40], 1, atol=2e-6)

def test_output_is_the_input_at_m_over_ratio(impl):
    """A tone in the pass band keeps its amplitude and phase, with no delay"""
    f, n = 0.31e6, 40000
    t = np.arange(n)/Fs
    x = np.exp(2j*np.pi*f*t + 0.4j).astype(np.complex64)
    y, _ = run(lambda: FractionalPolyphase(RATIO, cutoff=3.7e6), x, [n])
    want = np.exp(2j*np.pi*f*np.arange(len(y))/(Fs*RATIO) + 0.4j)
    np.testing.assert_allclose(y[64:-64], want[64:-64], atol=2e-3) # the phase grid is 1/1024 sample

def test_matches_resample_poly_for_a_rational_ratio(impl):
    x = noise(6000)
    want = resample_poly(x, 3, 4)
    got = fractional_resample(x, 0.75, Fs=1.0)
    n = min(len(want), len(got)) - 80
    assert np.abs(got[40:n] - want[40:n]).max() < 0.02*np.abs(want).max() # two different windowed sincs agree within their stop bands

def test_alias_rejection(impl):
    """Decimating by 4 with the default cutoff: a tone above the new Nyquist is rejected to the design depth"""
    n = 20000
    x = np.exp(2j*np.pi*0.30*np.arange(n)).astype(np.complex64) # 0.30 cycles/sample > 0.125 (the output Nyquist)
    y, _ = run(lambda: FractionalPolyphase(0.25), x, [n], dict(Fs=1.0, sps=1))
    assert 20*np.log10(np.abs(y[40:-40]).max()) < -60

def test_fused_shift_is_shift_then_resample(impl):
    x = noise(30000)
    fused, _ = run(lambda: FractionalPolyphase(RATIO, cutoff=3.7e6, shift=-0.5e6), x, [len(x)])
    shifted, _ = run(lambda: rp.signal.filters.ShiftFreq(-0.5e6), x, [len(x)])
    seq, _ = run(lambda: FractionalPolyphase(RATIO, cutoff=3.7e6), shifted, [len(shifted)])
    np.testing.assert_allclose(fused, seq, atol=2e-4)

def test_ratio_can_be_trimmed_between_blocks(impl):
    """A ratio change is continuous: no repeated or dropped input, the count follows the ratio, a tone stays a tone"""
    n, f = 60000, 0.011
    x = np.exp(2j*np.pi*f*np.arange(n)).astype(np.complex64)
    stage = FractionalPolyphase(0.5)
    stream, outs, i = None, [], 0
    for k, b in enumerate([20000]*3):
        stage.ratio = 0.5 if k == 0 else 0.5*(1 + 20e-6*k) # a 20 ppm trim
        d = rp.Data(x[i:i + b], Fs=1.0, sps=1, stream=stream)
        stream = d.stream
        outs.append(np.asarray((d + stage).data))
        i += b
    y = np.concatenate(outs)
    assert abs(len(y) - 0.5*n) < 12 # the last taps/2 input samples are still held
    ph = np.unwrap(np.angle(y[100:-100]*np.exp(-2j*np.pi*(f*2)*np.arange(100, len(y) - 100))))
    inc = np.diff(ph)
    assert np.abs(inc - np.median(inc)).max() < 2e-3 # no jump in the tone's phase where the ratio changed (the trim itself moves it ~1e-5 rad a sample)

def test_zero_length_blocks_and_short_first_block(impl):
    y, _ = run(lambda: FractionalPolyphase(RATIO), noise(100), [0, 3, 0, 40, 57], dict(Fs=Fs, sps=1))
    ref, _ = run(lambda: FractionalPolyphase(RATIO), noise(100), [100], dict(Fs=Fs, sps=1))
    np.testing.assert_allclose(y, ref, atol=2e-5)

def test_settings_are_checked():
    for bad in (dict(ratio=0), dict(ratio=1, taps=7)):
        with pytest.raises(ValueError):
            FractionalPolyphase(**bad)
    with pytest.raises(ValueError):
        design_bank(32, 512, 0.7)

def test_bank_rows_have_unit_dc_gain():
    t = design_bank(32, 512, 0.185)
    assert t.shape == (513, 32) and np.allclose(t.sum(axis=1), 1)
    np.testing.assert_allclose(t[-1][1:], t[0][:-1], atol=1e-7) # a whole sample past is the first row one sample on: no jump where the position wraps
    assert t[0][-1] == 0 and t[-1][0] == 0

@pytest.mark.skipif(_accel.native is None, reason="rpps._native not built")
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

@pytest.mark.skipif(_accel.native is None, reason="rpps._native not built")
def test_native_rotate_matches_numpy():
    x = noise(100000)
    out = np.empty_like(x)
    w = 2*np.pi*-0.5e6/20e6
    ph = _accel.native.rotate_into(x, out, 1.3, w, 0)
    np.testing.assert_allclose(out, x*np.exp(1j*(1.3 + w*np.arange(len(x)))), atol=2e-6)
    assert ph == pytest.approx((1.3 + w*len(x)) % (2*np.pi))
