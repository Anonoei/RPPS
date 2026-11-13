"""Parks-McClellan design (rpps.signal.filters.impl.remez): the equiripple property itself, then scipy as an oracle when it is installed"""
import numpy as np
import pytest

from rpps.signal.filters.impl.remez import remez

def _amp(h, f, antisym):
    """Real amplitude response at frequencies f (fractions of fs) of a linear-phase filter"""
    n = len(h); c = (n - 1)/2
    ph = np.exp(-2j*np.pi*np.outer(f, np.arange(n) - c))
    H = ph @ h
    return (-1j*H).real if antisym else H.real # scipy's odd types: H = +j A, negative impulse response after the centre

def _errors(h, bands, desired, weight, kind="bandpass"):
    """Weighted error of each band on a dense grid"""
    out = []
    for (lo, hi), d, w in zip(np.reshape(bands, (-1, 2)), desired, weight):
        f = np.linspace(lo, hi, 4000)
        target = d*f if kind == "differentiator" else d
        wt = w/f if kind == "differentiator" and d else w
        out.append(wt*(target - _amp(h, f, kind != "bandpass")))
    return out

CASES = [
    (41, [0, .1, .15, .5], [1, 0], None),
    (40, [0, .1, .15, .5], [1, 0], None),
    (51, [0, .08, .12, .3, .35, .5], [0, 1, 0], [10, 1, 10]),
    (101, [0, .1, .15, .25, .3, .5], [0, 1, 0], None),
    (301, [0, .1, .12, .5], [1, 0], [1, 50]),
]

@pytest.mark.parametrize("n,bands,desired,weight", CASES)
def test_error_equioscillates(n, bands, desired, weight):
    """Optimal: the weighted error reaches the same peak in every band that has the weight to carry it, and the response is symmetric"""
    h = remez(n, bands, desired, weight)
    assert np.allclose(h, h[::-1])
    w = np.ones(len(desired)) if weight is None else np.asarray(weight, float)
    peaks = [np.abs(e).max() for e in _errors(h, bands, desired, w)]
    assert max(peaks)/min(peaks) < 1.02 # each band is held to the same weighted ripple

@pytest.mark.parametrize("kind,n", [("hilbert", 41), ("hilbert", 40), ("differentiator", 41), ("differentiator", 40)])
def test_odd_symmetry_types(kind, n):
    h = remez(n, [.05, .45], [1], type=kind)
    assert np.allclose(h, -h[::-1])
    if kind == "hilbert":
        assert h[n//2 + 1 if n % 2 else n//2] < 0 # scipy's sign: a negative tail after the centre
        f = np.linspace(.1, .4, 50)
        assert np.abs(_amp(h, f, True) - 1).max() < 0.02
    else:
        f = np.linspace(.1, .4, 50)
        assert np.abs(_amp(h, f, True)/f - 1).max() < 0.02

def test_hz_and_fs():
    a = remez(41, [0, 1000, 1500, 5000], [1, 0], fs=10e3)
    b = remez(41, [0, .1, .15, .5], [1, 0])
    assert np.allclose(a, b)

def test_bad_specs():
    with pytest.raises(ValueError):
        remez(41, [0, .1, .15], [1, 0])
    with pytest.raises(ValueError):
        remez(41, [0, .1, .15, .5], [1])
    with pytest.raises(ValueError):
        remez(41, [0, .1, .15, .6], [1, 0])
    with pytest.raises(ValueError):
        remez(41, [0, .1, .15, .5], [1, 0], weight=[1, -1])
    with pytest.raises(ValueError):
        remez(41, [0, .1, .15, .5], [1, 0], type="nope")

# ---- Against scipy, when it is there

@pytest.mark.parametrize("n,bands,desired,weight", CASES)
def test_matches_scipy(n, bands, desired, weight):
    ss = pytest.importorskip("scipy.signal")
    ref = ss.remez(n, bands, desired, weight=weight, fs=1.0)
    assert np.abs(remez(n, bands, desired, weight) - ref).max() < 1e-3*np.abs(ref).max()

@pytest.mark.parametrize("kind,n", [("hilbert", 41), ("hilbert", 40), ("differentiator", 41), ("differentiator", 40)])
def test_odd_types_match_scipy(kind, n):
    ss = pytest.importorskip("scipy.signal")
    assert np.allclose(remez(n, [.05, .45], [1], type=kind), ss.remez(n, [.05, .45], [1], type=kind, fs=1.0), atol=1e-4)
