"""The array helpers that replace scipy in the examples (resample_poly, convolve, percentile_filter, least_squares), each checked against
scipy when it is installed (the oracle) and against its definition when it is not"""
import ast
import pathlib

import numpy as np
import pytest

import rpps as rp

@pytest.fixture
def rng():
    return np.random.default_rng(7)

def test_rpps_never_imports_scipy():
    """scipy is a test oracle only: nothing under src/ may import it"""
    for path in pathlib.Path(rp.__file__).parent.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""] if isinstance(node, ast.ImportFrom) else []
            assert not any(n.split(".")[0] == "scipy" for n in names), f"{path} imports scipy"

# ---- resample_poly

RATES = [(1, 5), (2, 3), (3, 2), (48, 125), (7, 1), (147, 160), (1, 625)]

@pytest.mark.parametrize("up,down", RATES)
def test_resample_poly_length_and_tone(up, down, rng):
    n = 20000 + 80*max(up, down) # long enough that the middle is clear of the filter edges
    f = 0.2*min(1, up/down) # a tone well inside the lower rate's band
    x = np.exp(2j*np.pi*f*np.arange(n))
    y = rp.signal.sample.resample_poly(x, up, down)
    assert len(y) == -(-n*up//down)
    m = np.arange(len(y))
    mid = slice(len(y)//4, 3*len(y)//4) # clear of the edges
    assert np.allclose(y[mid], np.exp(2j*np.pi*f*m*down/up)[mid], atol=2e-3) # zero delay: output m is the input at m*down/up

def test_resample_poly_dtypes_and_identity(rng):
    x = rng.standard_normal(1000).astype(np.float32)
    assert rp.signal.sample.resample_poly(x, 1, 4).dtype == np.float32
    assert rp.signal.sample.resample_poly(x.astype(np.complex64), 1, 4).dtype == np.complex64
    assert rp.signal.sample.resample_poly(x.astype(np.float64), 1, 4).dtype == np.float64
    assert np.array_equal(rp.signal.sample.resample_poly(x, 3, 3), x)
    assert len(rp.signal.sample.resample_poly(x[:0], 1, 4)) == 0
    with pytest.raises(ValueError):
        rp.signal.sample.resample_poly(x, 0, 4)

@pytest.mark.parametrize("up,down", RATES)
@pytest.mark.parametrize("cplx", [False, True])
def test_resample_poly_matches_scipy(up, down, cplx, rng):
    ss = pytest.importorskip("scipy.signal")
    x = rng.standard_normal(6001) + (1j*rng.standard_normal(6001) if cplx else 0)
    assert np.allclose(rp.signal.sample.resample_poly(x, up, down), ss.resample_poly(x, up, down), atol=1e-6)
    assert np.allclose(rp.signal.sample.resample_poly(x, up, down, window=("kaiser", 8.0)), ss.resample_poly(x, up, down, window=("kaiser", 8.0)), atol=1e-6)

# ---- convolve

@pytest.mark.parametrize("mode", ["full", "same", "valid"])
@pytest.mark.parametrize("nx,nh", [(100, 7), (50001, 17), (100000, 401), (30000, 64), (9000, 3000)])
@pytest.mark.parametrize("cplx", [False, True])
def test_convolve_matches_numpy(mode, nx, nh, cplx, rng):
    x = rng.standard_normal(nx) + (1j*rng.standard_normal(nx) if cplx else 0)
    h = rng.standard_normal(nh)
    out = rp.signal.filters.convolve(x, h, mode)
    ref = np.convolve(x, h, mode)
    assert out.shape == ref.shape and np.allclose(out, ref, atol=1e-9)

def test_convolve_single_precision_and_errors(rng):
    x = rng.standard_normal(20000).astype(np.float32); h = rng.standard_normal(101).astype(np.float32)
    assert rp.signal.filters.convolve(x, h).dtype == np.float32
    with pytest.raises(ValueError):
        rp.signal.filters.convolve(x, h, "nope")

def test_convolve_matches_scipy(rng):
    ss = pytest.importorskip("scipy.signal")
    x = rng.standard_normal(100000); h = rng.standard_normal(401)
    for mode in ("full", "same"):
        assert np.allclose(rp.signal.filters.convolve(x, h, mode), ss.oaconvolve(x, h, mode=mode), atol=1e-9)

# ---- percentile_filter

def _rank_ref(x, q, size, mode):
    lo, hi = size//2, (size - 1)//2
    xp = np.pad(x, (lo, hi), mode={"nearest": "edge", "reflect": "symmetric", "mirror": "reflect", "wrap": "wrap", "constant": "constant"}[mode])
    r = min(int(size*q/100), size - 1)
    return np.array([np.sort(xp[i:i + size])[r] for i in range(len(x))])

@pytest.mark.parametrize("q,size,mode", [(20, 101, "nearest"), (50, 10, "nearest"), (95, 7, "reflect"), (15, 1, "nearest"), (0, 5, "mirror"), (100, 6, "wrap"), (30, 9, "constant")])
def test_percentile_filter_definition(q, size, mode, rng):
    x = rng.standard_normal(400)
    assert np.array_equal(rp.math.percentile_filter(x, q, size, mode), _rank_ref(x, q, size, mode))

def test_percentile_filter_short_input_and_errors(rng):
    x = rng.standard_normal(30)
    assert np.array_equal(rp.math.percentile_filter(x, 20, 101), _rank_ref(x, 20, 101, "nearest")) # window longer than the data
    assert len(rp.math.percentile_filter(x[:0], 20, 5)) == 0
    for bad in (dict(percentile=120, size=5), dict(percentile=20, size=0), dict(percentile=20, size=5, mode="x")):
        with pytest.raises(ValueError):
            rp.math.percentile_filter(x, **bad)

def test_percentile_filter_matches_scipy(rng):
    nd = pytest.importorskip("scipy.ndimage")
    x = rng.standard_normal(5000)
    for q, size in ((20, 400), (15, 1001), (50, 8)):
        assert np.array_equal(rp.math.percentile_filter(x, q, size), nd.percentile_filter(x, q, size=size, mode="nearest"))

# ---- least_squares

def _fix_problem():
    rx = np.array([[0, 0], [3000, 0], [0, 4000], [5000, 5000.]]); p0 = np.array([1200., 2100.])
    d = np.linalg.norm(rx - p0, axis=1)
    def res(p):
        dd = np.linalg.norm(rx - p, axis=1)
        return ((dd[1:] - dd[0]) - (d[1:] - d[0]))/1e-3
    return res, p0

def test_least_squares_position_fix():
    res, p0 = _fix_problem()
    r = rp.math.least_squares(res, np.array([1500., 1500.]))
    assert r.success and np.allclose(r.x, p0, atol=1e-3) and r.cost < 1e-6
    assert r.jac.shape == (3, 2) and r.fun.shape == (3,)

def test_least_squares_curve_fit_with_noise(rng):
    t = np.linspace(0, 1, 200); y = 2.5*np.exp(-1.3*t) + 0.01*rng.standard_normal(200)
    r = rp.math.least_squares(lambda p: p[0]*np.exp(-p[1]*t) - y, [1.0, 1.0])
    assert np.allclose(r.x, [2.5, 1.3], atol=0.05)

def test_least_squares_analytic_jacobian():
    t = np.linspace(0, 1, 50); y = 3*t + 1
    r = rp.math.least_squares(lambda p: p[0]*t + p[1] - y, [0., 0.], jac=lambda p: np.stack([t, np.ones_like(t)], 1))
    assert np.allclose(r.x, [3, 1], atol=1e-8)

def test_least_squares_matches_scipy():
    so = pytest.importorskip("scipy.optimize")
    res, _ = _fix_problem()
    assert np.allclose(rp.math.least_squares(res, np.array([4000., 100.])).x, so.least_squares(res, np.array([4000., 100.])).x, atol=1e-3)
