"""The front of the receiver: pieces made cheaper keep their results"""
import numpy as np

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
