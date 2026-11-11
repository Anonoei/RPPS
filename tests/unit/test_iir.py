import json
import warnings
import numpy as np
import pytest

import rpps as rp
from rpps import _config
from rpps.errors import ConfigError
from rpps.signal.filters import iir

def lfilter_ref(b, a, x):
    """y[n] = (sum b[k] x[n-k] - sum a[k] y[n-k]) / a[0], one sample at a time"""
    b, a = np.asarray(b, float), np.asarray(a, float)
    y = np.zeros(len(x), dtype=np.result_type(x, float))
    for n in range(len(x)):
        acc = sum(b[k]*x[n-k] for k in range(len(b)) if n-k >= 0) - sum(a[k]*y[n-k] for k in range(1, len(a)) if n-k >= 0)
        y[n] = acc / a[0]
    return y

def gain_db(sos, f):
    """Magnitude (dB) at the frequency f (cycles per sample)"""
    z = np.exp(-2j*np.pi*f)
    H = np.prod([(b0 + b1*z + b2*z**2)/(a0 + a1*z + a2*z**2) for b0, b1, b2, a0, a1, a2 in iir.as_sos(sos)])
    with np.errstate(divide="ignore"):
        return 20*np.log10(abs(H))

@pytest.fixture
def x():
    rng = np.random.default_rng(7)
    return rng.normal(size=3000) + 1j*rng.normal(size=3000)

# ---- The stage

def test_stage_matches_the_recurrence(x):
    sos = iir.design("butter", 4, "lowpass", 0.15)
    ref = x.copy()
    for row in sos:
        ref = lfilter_ref(row[:3], row[3:], ref)
    st = iir.IIR(sos)
    assert np.abs(st.run(x) - ref).max() < 1e-9
    assert np.abs(st.run(x.real) - ref.real).max() < 1e-9 # Real input stays real
    assert st.run(x.real).dtype.kind == "f"

def test_any_block_split_gives_the_same_output(x):
    st = iir.IIR(iir.design("cheby1", 5, "bandpass", [0.1, 0.2], ripple_db=1))
    whole = st.run(x)
    z, out, i = st.new_state(None), [], 0
    for n in (0, 1, 7, 255, 256, 257, 300, 0, 1000, 3):
        out.append(st.step(x[i:i+n], z, None))
        i += n
    out.append(st.step(x[i:], z, None))
    assert np.abs(np.concatenate(out) - whole).max() < 1e-10

def test_through_a_stream_and_reusable(x):
    st = rp.signal.filters.IIR(iir.design("butter", 3, "lowpass", 0.1))
    s = x.astype(np.complex64)
    a = rp.Data(s, Fs=1e3) + st
    b = rp.Data(s, Fs=1e3) + st # A new stream starts from zero state
    assert np.array_equal(a.data, b.data) and a.data.dtype == np.complex64
    parts, stream = [], None
    for i in range(0, 3000, 1000):
        d = rp.Data(s[i:i+1000], Fs=1e3, stream=stream) + st
        stream = d.stream
        parts.append(d.data)
    assert np.allclose(np.concatenate(parts), a.data, atol=1e-5)

def test_unstable_rejected_integrator_allowed():
    with pytest.raises(ValueError, match="unstable"):
        iir.IIR([[1, 0, 0, 1, -1.01, 0]])
    acc = iir.IIR([[1, 0, 0, 1, -1, 0]]).run(np.ones(10))
    assert np.allclose(acc, np.arange(1, 11))

def test_gain_and_validation():
    assert np.allclose(iir.IIR([[1, 0, 0, 1, 0, 0]], gain=3).run(np.ones(4)), 3)
    for bad in ([[1, 2, 3]], [[1, 0, 0, 0, 0, 0]], []):
        with pytest.raises(ValueError):
            iir.as_sos(bad)

# ---- Designs, checked by their defining properties

@pytest.mark.parametrize("n", [1, 2, 3, 4, 7])
def test_butterworth(n):
    lp = iir.design("butter", n, "lowpass", 0.1)
    assert gain_db(lp, 0.0) == pytest.approx(0.0, abs=1e-9) and gain_db(lp, 0.1) == pytest.approx(-3.0103, abs=1e-6)
    hp = iir.design("butter", n, "highpass", 0.1)
    assert gain_db(hp, 0.5) == pytest.approx(0.0, abs=1e-9) and gain_db(hp, 0.1) == pytest.approx(-3.0103, abs=1e-6)
    assert gain_db(lp, 0.3) < -20*n*0.4 # Rolls off about 20 dB/decade per pole, at least
    assert len(poles := iir.poles(lp)) == n and np.all(np.abs(poles) < 1)

@pytest.mark.parametrize("n", [3, 4, 6])
def test_chebyshev1_ripple(n):
    sos = iir.design("cheby1", n, "lowpass", 0.2, ripple_db=1.5)
    g = np.array([gain_db(sos, f) for f in np.linspace(0, 0.2, 400)])
    assert g.max() <= 1e-9 and g.min() >= -1.5 - 1e-6 and g[-1] == pytest.approx(-1.5, abs=1e-6)
    assert gain_db(sos, 0) == pytest.approx(-1.5 if n % 2 == 0 else 0.0, abs=1e-6) # Even orders start in a trough

@pytest.mark.parametrize("n", [3, 4, 7])
def test_chebyshev2_stopband(n):
    sos = iir.design("cheby2", n, "lowpass", 0.2, atten_db=45)
    assert gain_db(sos, 0) == pytest.approx(0.0, abs=1e-9)
    assert max(gain_db(sos, f) for f in np.linspace(0.2, 0.5, 300)) <= -45 + 1e-6
    assert gain_db(sos, 0.2) == pytest.approx(-45, abs=1e-6)

@pytest.mark.parametrize("kind", ["butter", "cheby1", "cheby2"])
def test_band_designs(kind):
    kw = dict(ripple_db=1.0, atten_db=40.0)
    bp = iir.design(kind, 3, "bandpass", [0.15, 0.25], **kw)
    bs = iir.design(kind, 3, "bandstop", [0.15, 0.25], **kw)
    assert len(iir.poles(bp)) == len(iir.poles(bs)) == 6
    assert gain_db(bp, 0.0) < -30 and gain_db(bp, 0.5) < -30 and gain_db(bp, 0.2) > -3.5
    assert gain_db(bs, 0.0) > -3.5 and gain_db(bs, 0.5) > -3.5 and gain_db(bs, 0.2) < -25

def test_high_order_stays_stable_and_accurate():
    sos = iir.design("butter", 16, "lowpass", 0.05)
    assert np.all(np.abs(iir.poles(sos)) < 1)
    assert gain_db(sos, 0.05) == pytest.approx(-3.0103, abs=1e-5)
    y = iir.IIR(sos).run(np.ones(4000))
    assert y[-1] == pytest.approx(1.0, abs=1e-6)

def test_design_arguments_are_checked():
    for args in (("butter", 0, "lowpass", 0.1), ("butter", 2, "lowpass", 0.5), ("butter", 2, "lowpass", 0.0),
                 ("butter", 2, "bandpass", 0.1), ("butter", 2, "bandpass", [0.3, 0.2]), ("butter", 2, "sideways", 0.1),
                 ("cheby1", 2, "lowpass", 0.1), ("cheby2", 2, "lowpass", 0.1), ("elliptic", 2, "lowpass", 0.1)):
        with pytest.raises(ValueError):
            iir.design(*args)

def test_rbj_biquads():
    f0 = 0.1
    assert gain_db(iir.biquad("peaking", f0, 2, gain_db=6), f0) == pytest.approx(6, abs=1e-9)
    assert gain_db(iir.biquad("peaking", f0, 2, gain_db=6), 0.0) == pytest.approx(0, abs=1e-9)
    assert gain_db(iir.biquad("notch", f0, 10), f0) < -200
    assert gain_db(iir.biquad("lowshelf", f0, gain_db=-9), 0.0) == pytest.approx(-9, abs=1e-9)
    assert gain_db(iir.biquad("highshelf", f0, gain_db=5), 0.5) == pytest.approx(5, abs=1e-9)
    assert gain_db(iir.biquad("lowpass", f0), 0.0) == pytest.approx(0, abs=1e-9)
    assert gain_db(iir.biquad("highpass", f0), 0.5) == pytest.approx(0, abs=1e-9)
    assert gain_db(iir.biquad("bandpass", f0, 4), f0) == pytest.approx(0, abs=1e-9)
    assert all(gain_db(iir.biquad("allpass", f0, 3), f) == pytest.approx(0, abs=1e-9) for f in (0.0, 0.05, 0.3))
    for bad in (("wobble", 0.1), ("notch", 0.0), ("notch", 0.5), ("notch", 0.1, -1)):
        with pytest.raises(ValueError):
            iir.biquad(*bad)

def test_single_pole():
    y = iir.IIR(iir.single_pole(0.2)).run(np.ones(60))
    assert y[0] == pytest.approx(0.2) and y[-1] == pytest.approx(1.0, abs=1e-5)
    assert np.allclose(y, lfilter_ref([0.2], [1, -0.8], np.ones(60)))
    for bad in (0, -0.1, 1.5):
        with pytest.raises(ValueError):
            iir.single_pole(bad)

# ---- Sections from coefficients

@pytest.mark.parametrize("b,a", [([0, 0, 1], [1, -.5]), ([1, 2, 3, 4], [1]), ([2], [1, -.3]), ([0, 1.0], [1, 0.2, 0.5]),
                                 ([1, .5, .25, .1, .05], [1, -1.2, 0.9, -.2]), ([3.0], [2.0, -1.0])])
def test_tf2sos_matches_the_recurrence(b, a, x):
    xr = x.real[:400]
    assert np.abs(iir.IIR(iir.tf2sos(b, a)).run(xr) - lfilter_ref(b, a, xr)).max() < 1e-10

def test_zpk2sos_fewer_zeros_means_delay():
    xr = np.random.default_rng(1).normal(size=200)
    assert np.abs(iir.IIR(iir.zpk2sos([], [0.5], 1.0)).run(xr) - lfilter_ref([0, 1], [1, -.5], xr)).max() < 1e-12
    with pytest.raises(ValueError, match="causal"):
        iir.zpk2sos([0.1, 0.2], [0.5], 1.0)

def test_tf2sos_errors_and_high_order_warning():
    with pytest.raises(ValueError):
        iir.tf2sos([0, 0], [1])
    with pytest.raises(ValueError):
        iir.tf2sos([1], [0, 1])
    with pytest.warns(UserWarning, match="order"):
        iir.tf2sos(np.poly(np.full(10, 0.5)), [1])

# ---- Against scipy, when it is there

@pytest.mark.parametrize("kind", ["butter", "cheby1", "cheby2"])
@pytest.mark.parametrize("pass_,cut", [("lowpass", 0.12), ("highpass", 0.12), ("bandpass", [0.1, 0.2]), ("bandstop", [0.1, 0.2])])
@pytest.mark.parametrize("n", [1, 2, 5, 8])
def test_designs_match_scipy(kind, pass_, cut, n, x):
    ss = pytest.importorskip("scipy.signal")
    mine = iir.IIR(iir.design(kind, n, pass_, cut, ripple_db=1.0, atten_db=40.0))
    extra = {"butter": [], "cheby1": [1.0], "cheby2": [40.0]}[kind]
    ref = getattr(ss, kind)(n, *extra, np.array(cut)*2, btype=pass_, output="sos")
    f, H = mine.response(512)
    assert np.abs(H - ss.sosfreqz(ref, worN=2*np.pi*f)[1]).max() < 1e-9
    assert np.abs(mine.run(x) - ss.sosfilt(ref, x)).max() < 1e-8 * max(1, np.abs(ss.sosfilt(ref, x)).max())
