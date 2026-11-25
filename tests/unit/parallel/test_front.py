"""The front of the receiver: pieces made cheaper keep their results"""
import numpy as np
import pytest

import rpps as rp
from support.dvbs2 import KNOWN

# ---- the parallel runner on DVB-S2: the same output as the serial runner

# ---- the front: pieces that were made cheaper keep their results

def test_overlap_save_convolution_equals_np_convolve():
    from rpps.signal.filters.convolve import convolve
    rng = np.random.default_rng(0)
    for nx in (4096, 5000, 100003):
        for nh in (16, 26, 101, 513):
            for cplx in (False, True):
                x = rng.normal(size=nx) + (1j*rng.normal(size=nx) if cplx else 0)
                h = rng.normal(size=nh)
                for mode in ("full", "same", "valid"):
                    a, b = np.convolve(x, h, mode), convolve(x, h, mode)
                    assert a.shape == b.shape and np.allclose(a, b, atol=1e-9), (nx, nh, cplx, mode)

def test_phase_line_phasor_equals_the_exponential():
    from rpps.sync import track
    for length in (1, 7, 1000, 33333):
        for rate in (0.0, 1e-4, -3e-3):
            want = np.exp(-1j*(0.3 + rate*np.arange(length)))
            np.testing.assert_allclose(track.line_phasor(0.3, rate, length), want, atol=1e-12)
    assert len(track.line_phasor(0.0, 0.1, 0)) == 0

def test_power_by_squaring():
    from rpps.sync.phase.mpower import _power
    rng = np.random.default_rng(1)
    u = np.exp(1j*rng.uniform(-np.pi, np.pi, 500))
    for M in (1, 2, 3, 4, 8, 12):
        np.testing.assert_allclose(_power(u, M), u**M, atol=1e-12)

def test_the_matched_filter_gives_symbols_back_at_their_amplitude():
    """The profile's pulse shaping then its matched filter: unit symbols are about unit at their sampling instants, so a
    demapper sees the constellation's scale (they came back a quarter of it, at 4 samples per symbol)"""
    p = rp.profile.load("dvbs2/normal", sync={"timing": {"method": "known", "fuse": False}})
    syms = np.exp(1j*(np.pi/4 + np.pi/2*np.random.default_rng(2).integers(0, 4, 600))).astype(np.complex64)
    matched = p.rx().ops[0][1]
    pulse = p.tx().ops[-1][1]
    y = np.asarray((rp.Data(syms, Fs=1e6, sps=1, flush=True) + rp.signal.sample.up.Zeros(4) + pulse + matched).data)
    medians = [np.median(np.abs(y[k + 400:-400:4])) for k in range(4)]
    assert 0.9 < max(medians) < 1.1

def test_native_word_metric_equals_the_numpy_one(monkeypatch):
    """FrameSync's header-word metric in C++ and in numpy: the same frames found, metrics equal to single precision"""
    from rpps import _accel
    if _accel.native is None:
        pytest.skip("rpps._native is not built")
    p = rp.profile.load("dvbs2/normal", sync=KNOWN)
    fs = next(op for _, op in p.rx().ops if type(op).__name__ == "FrameSync")
    rng = np.random.default_rng(3)
    y = (rng.normal(size=20000) + 1j*rng.normal(size=20000)).astype(np.complex128)
    y[5000:5000 + fs.L] = fs.preamble*0.8
    c = np.convolve(y, np.conj(fs.preamble[::-1]), "valid")
    cs = np.concatenate(([0.0], np.cumsum(y.real**2 + y.imag**2)))
    m = np.abs(c)/np.sqrt(np.maximum(cs[fs.L:fs.L + len(c)] - cs[:len(c)], 1e-30)*fs._ep)
    a = fs._with_words(y, c, m, cs)
    monkeypatch.setattr(_accel, "native", None)
    b = fs._with_words(y, c, m, cs)
    np.testing.assert_allclose(a, b, rtol=1e-4, atol=1e-5)
    assert (a > 0).sum() > 100

def test_fused_matched_filter_gives_the_same_symbols_as_filtering_every_sample():
    """Known timing with the matched filter evaluated only at its instants, against the filter then the pick, for any block cut"""
    rng = np.random.default_rng(4)
    for name, kw in (("dvbs2/normal", {}), ("ieee802154/2450", {"sps": 4})):
        fused = rp.profile.load(name, sync=KNOWN, **kw)
        apart = rp.profile.load(name, sync={"timing": {"method": "known", "fuse": False}}, **kw)
        x = (rng.normal(size=40000) + 1j*rng.normal(size=40000)).astype(np.complex64)
        def front(p, cuts):
            ops = [op for _, op in p.rx().ops]
            stop = next(i for i, o in enumerate(ops) if isinstance(o, rp.sync.time.Known)) + 1
            st, outs, i = None, [], 0
            for k, n in enumerate(cuts):
                d = rp.Data(x[i:i + n], Fs=1e6, sps=4, stream=st, flush=k == len(cuts) - 1)
                st, i = d.stream, i + n
                for o in ops[:stop]:
                    d = d + o
                outs.append(np.asarray(d.data))
            return np.concatenate(outs)
        want = front(apart, [40000])
        for cuts in ([40000], [10000]*4, [1, 7, 12345, 40000 - 12353]):
            got = front(fused, cuts)
            assert got.shape == want.shape, (name, cuts)
            np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-5)

def test_native_carrier_helpers_equal_the_numpy_ones(monkeypatch):
    from rpps import _accel
    from rpps.sync.phase import mpower
    if _accel.native is None:
        pytest.skip("rpps._native is not built")
    rng = np.random.default_rng(6)
    seg = (rng.normal(size=1500) + 1j*rng.normal(size=1500))
    seg[::37] = 0.0 # zeros stay zero
    for M, phi, rate in ((8, 0.4, 3e-4), (4, -2.0, -1e-3), (2, 0.0, 0.0), (1, 1.0, 5e-3)):
        a = mpower._measure(seg, phi, rate, M)
        r = mpower._turn_back(seg, phi, rate)
        monkeypatch.setattr(_accel, "native", None)
        b = mpower._measure(seg, phi, rate, M)
        q = mpower._turn_back(seg, phi, rate)
        monkeypatch.undo()
        np.testing.assert_allclose(a[0], b[0], rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(a[1], b[1], rtol=1e-9)
        np.testing.assert_allclose(r, q, rtol=1e-5, atol=1e-6)

def test_native_preamble_metric_equals_the_numpy_one():
    from rpps import _accel
    if _accel.native is None:
        pytest.skip("rpps._native is not built")
    rng = np.random.default_rng(7)
    for L in (1, 7, 26, 255):
        y = rng.normal(size=20000) + 1j*rng.normal(size=20000)
        y[3000:3000 + L] *= 4
        pre = np.exp(2j*np.pi*rng.random(L))
        c, m = _accel.native.preamble_metric(y, pre, float(np.sum(np.abs(pre)**2)))
        want_c = np.correlate(y, pre, "valid")
        cs = np.concatenate(([0.0], np.cumsum(np.abs(y)**2)))
        e = cs[L:] - cs[:-L]
        want_m = np.abs(want_c)/np.sqrt(np.maximum(e, 1e-30)*np.sum(np.abs(pre)**2))
        np.testing.assert_allclose(c, want_c, rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(m, want_m, rtol=1e-8, atol=1e-9)
