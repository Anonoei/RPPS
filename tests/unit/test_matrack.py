"""MatTrack (`timing: mat`): the matched filter and timing in one, on numpy and on a GPU"""
from types import SimpleNamespace

import numpy as np
import pytest

import rpps as rp
from rpps import _accel
from rpps.math.interp import interpolate
from rpps.sync.time import MatTrack

XP, BACKEND = _accel.device()
GPU = pytest.mark.skipif(BACKEND is None, reason="no GPU (rpps._accel.device())")
FRAMES = [{"modcod": (4, 12)[i % 2], "pilots": 1} for i in range(4)]
MAT = {"timing": {"method": "mat"}}
MAT_DEV = {"timing": {"method": "mat", "device": True}}

@pytest.fixture(scope="module")
def s2():
    """DVB-S2 frames after a quiet start, received with a clock 20 ppm fast, a phase offset and noise"""
    p = rp.profile.load("dvbs2/normal")
    t = rp.profile.table("dvbs2/modcod")
    dfl = [t.row({**f, "short": 0})["kbch"] - 80 for f in FRAMES]
    data = np.random.default_rng(1).integers(0, 2, sum(dfl)).astype(bool)
    o = p.tx()(rp.Data(data, Fs=1e6, sps=1, flush=True), fields=FRAMES)
    x = np.concatenate((np.zeros(3000, np.complex64), o.data, np.zeros(400, np.complex64)))
    x = interpolate(x, np.arange(int(len(x)/(1 + 20e-6)) - 20)*(1 + 20e-6))
    rng = np.random.default_rng(2)
    x = (x*np.exp(0.7j) + 0.05*(rng.normal(size=len(x)) + 1j*rng.normal(size=len(x)))).astype(np.complex64)
    return x, o, data

def receive(x, o, sync, cut=None, parallel=None):
    rx = rp.profile.load("dvbs2/normal", sync=sync).rx(parallel=parallel)
    st, outs = None, []
    cut = cut or len(x)
    for a in range(0, len(x), cut):
        d = rp.Data(x[a:a + cut], Fs=o.Fs, sps=4, delay=o.delay, stream=st, flush=a + cut >= len(x))
        st = d.stream
        outs.append(np.asarray(rx(d).data).astype(bool))
    if parallel is not None:
        rx.close()
    return np.concatenate(outs), st

def test_the_profile_fuses_its_matched_filter():
    p = rp.profile.load("dvbs2/normal", sync=MAT)
    steps = p._build()
    shape = next(s for s in steps if s.category == "wave" and s.kind == "shape")
    timing = next(s for s in steps if s.category == "sync" and s.kind == "timing")
    assert shape.rx == [] and str(timing.rx[0][1]) == "MatTrack(matched)"

def test_dvbs2_decodes_with_a_clock_off(s2):
    x, o, data = s2
    got, st = receive(x, o, MAT)
    assert np.array_equal(got, data)
    est = st.est["timing"]
    assert abs(est["ppm"] - 20) < 1 and est["missed"] == 0

def test_any_block_cut_gives_the_same(s2):
    x, o, data = s2
    whole, _ = receive(x, o, MAT)
    cut, _ = receive(x, o, MAT, cut=30011)
    assert np.array_equal(cut, whole)

def _held(n=200_000, seed=0, cplx=False):
    rng = np.random.default_rng(seed)
    h = np.hanning(33)
    h = (h + 1j*np.roll(h, 3)*0.3) if cplx else h
    return SimpleNamespace(buf=(rng.normal(size=n) + 1j*rng.normal(size=n)).astype(np.complex64), base=1000, taps=h/np.sum(np.abs(h)),
                           y=None, dy=None, dtaps=None)

@GPU
@pytest.mark.parametrize("cplx", [False, True])
def test_the_device_reads_what_the_reference_does(cplx):
    """Filtered on the device and read at fixed-point instants: the numpy reference's values to float32 rounding"""
    m = MatTrack(sps=4)
    tr = SimpleNamespace(value=1.3, rate=-2e-5, at=5000.0)
    sets = [m._window(tr, a, a + 8192, 4.0, off) for a in range(2000, 150_000, 8192) for off in (-1.0, 0.0, 1.0)]
    ref, dev = _held(cplx=cplx), _held(cplx=cplx)
    ref.xp, dev.xp = np, XP
    a, b = m._values(ref, sets, 4.0), m._values(dev, sets, 4.0)
    assert a.shape == b.shape and np.max(np.abs(a - b)) < 1e-4*np.sqrt(np.mean(np.abs(a)**2))
    np.testing.assert_allclose(m._energy(dev, sets, 4.0), m._energy(ref, sets, 4.0), rtol=1e-5)

@GPU
def test_on_the_device_too(s2):
    x, o, data = s2
    got, st = receive(x, o, MAT_DEV)
    assert np.array_equal(got, data)
    cut, _ = receive(x, o, MAT_DEV, cut=30011)
    assert np.array_equal(cut, got)

@GPU
def test_front_and_region_on_the_device(s2):
    x, o, data = s2
    got, _ = receive(x, o, MAT_DEV, parallel=rp.parallel.Parallel("thread", workers=2, xp=XP))
    assert np.array_equal(got, data)

@GPU
def test_frame_sync_on_the_device_finds_the_frames_the_host_does(s2):
    """The preamble and word metric on the device (single precision): the same frame starts and phases, so the same output"""
    from rpps.sync import FrameSync
    x, o, data = s2
    found = {}
    orig = FrameSync._peaks
    def spy(self, y, lo, hi):
        r = orig(self, y, lo, hi)
        found.setdefault(self.device, []).extend(r[0])
        return r
    FrameSync._peaks = spy
    try:
        host, _ = receive(x, o, MAT)
        dev, _ = receive(x, o, {**MAT, "frame": {"device": True}})
    finally:
        FrameSync._peaks = orig
    assert np.array_equal(host, data) and np.array_equal(dev, data)
    a, b = found[False], found[True]
    assert [p[0] for p in a] == [p[0] for p in b] and len(a) == len(FRAMES)
    assert max(abs(p[1] - q[1]) for p, q in zip(a, b)) < 1e-4 and max(abs(p[2] - q[2]) for p, q in zip(a, b)) < 1e-4
