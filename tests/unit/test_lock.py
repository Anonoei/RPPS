"""rpps.sync.Lock: a snapshot lock found once and followed block by block, consecutive or not; and best_tau's two ways"""
import numpy as np
import pytest

import rpps as rp
from rpps.sync.time.mat import best_tau, _best_tau_rows

def psk(n, sps, M=4, beta=0.35, tau=0.0, cfo=0.0, esn0_db=20.0, seed=0):
    """n samples of M-PSK through an RRC at a fractional sps, the first symbol at tau, a carrier offset [cycles/sample]"""
    rng = np.random.default_rng(seed)
    k = np.arange(-20, int(n/sps) + 20)
    sym = np.exp(2j*np.pi*rng.integers(0, M, len(k))/M + 1j*np.pi/M*(M == 4))
    t = np.arange(n)
    x = np.zeros(n, complex)
    L = int(np.ceil(10*sps))
    for kk, s in zip(k, sym):
        c = tau + kk*sps
        j0, j1 = int(max(0, np.floor(c - L))), int(min(n, np.ceil(c + L)))
        if j1 > j0:
            x[j0:j1] += s*rp.signal.filters.rrc_pulse((t[j0:j1] - c)/sps, beta)
    x *= np.exp(2j*np.pi*cfo*t)
    n0 = np.mean(np.abs(x)**2)*sps/10**(esn0_db/10)
    return x + np.sqrt(n0/2)*(rng.standard_normal(n) + 1j*rng.standard_normal(n))

@pytest.mark.parametrize("sps", [2.0, 3.37, 7.91])
def test_best_tau_by_convolution_matches_the_filter_matrix(sps):
    x = psk(30000, sps, tau=0.62*sps)
    for lo, hi in ((0, None), (5000, 12000), (20000, 30000)):
        a, ea = best_tau(x, sps, 0.35, 8, lo, hi)
        b, eb = _best_tau_rows(x, sps, 0.35, 8, lo, hi)
        assert abs((a - b + sps/2) % sps - sps/2) < 0.01*sps and ea == pytest.approx(eb, rel=1e-3)
        assert abs((a - 0.62*sps + sps/2) % sps - sps/2) < 0.02*sps

def test_best_tau_short_segment_takes_the_matrix():
    x = psk(150, 4.0, tau=1.0) # Under 4 spans + 8 symbols: the convolution's edges would be most of it
    assert best_tau(x, 4.0, 0.35) == _best_tau_rows(x, 4.0, 0.35)

def test_track_timing_corrects_the_rate():
    # The guess 30 ppm off: the timing walks 1.6 samples across x. (The walk is seen modulo a symbol, so it must stay under
    # half of one between the two ends: some 40 ppm here)
    sps = 4.0*(1 + 30e-6)
    x = psk(60000, sps, tau=1.1)
    s, tau = rp.sync.track_timing(x, 4.0, 0.35)
    assert abs(s/sps - 1) < 3e-6

def test_lock_finds_then_follows_disconnected_blocks():
    Fs, sps = 1e6, 5.3
    x = psk(200000, sps, tau=2.0, cfo=1e-3)
    lk = rp.sync.Lock.find(x[:20000], Fs)
    assert lk.det.mod.lower().startswith("qpsk") and abs(lk.det.baud/(Fs/sps) - 1) < 1e-3
    assert lk.stale is None and lk.quality.esn0_db > 15 and len(lk.symbols) == len(lk.centres) > 1000
    assert abs(lk.det.cfo - 1e-3*Fs) < 30
    # A block much later, not consecutive: no search, the same signal, on its own lattice
    nxt = lk.update(x[150000:160000])
    assert nxt is not lk and nxt.found is lk.det and nxt.stale is None
    assert abs(nxt.det.baud/(Fs/sps) - 1) < 1e-3 and nxt.quality.esn0_db > 15
    assert nxt.det.mod == lk.det.mod and nxt.det.beta == lk.det.beta
    # The lattice: centres land on the symbols (a matched filter there gives the constellation)
    assert np.std(np.abs(nxt.symbols)) < 0.2

def test_lock_goes_stale_when_the_signal_goes():
    Fs = 1e6
    lk = rp.sync.Lock.find(psk(20000, 4.0, tau=1.0), Fs)
    rng = np.random.default_rng(1)
    noise = rng.standard_normal(10000) + 1j*rng.standard_normal(10000)
    gone = lk.update(noise)
    assert gone.stale and "Es/N0" in gone.stale

def test_no_lock_says_why():
    rng = np.random.default_rng(2)
    with pytest.raises(rp.errors.NoLock, match="samples"):
        rp.sync.Lock.find(np.zeros(100, complex), 1e6)
    noise = rng.standard_normal(20000) + 1j*rng.standard_normal(20000)
    with pytest.raises(rp.errors.NoLock, match="no symbol rate") as e:
        rp.sync.Lock.find(noise, 1e6)
    assert e.value.line is not None
    lk = rp.sync.Lock.find(psk(20000, 4.0), 1e6)
    with pytest.raises(rp.errors.NoLock, match="fewer than"):
        lk.update(np.zeros(100, complex))

def test_lock_with_a_known_rate_needs_no_line():
    x = psk(20000, 4.0, tau=0.5)
    lk = rp.sync.Lock.find(x, 1e6, baud=250e3)
    assert lk.det.line is None and abs(lk.det.baud/250e3 - 1) < 1e-3

def test_refine_corrects_a_rate_update_cannot_follow():
    """A first lock on few samples can be well off in rate; refine searches it again on more signal (Acquire does, as
    its history grows), where update, following, would lose it"""
    Fs, sps = 1e6, 4.0
    x = psk(200000, sps, tau=1.0, cfo=1e-3)
    lk = rp.sync.Lock.find(x[:20000], Fs)
    off = rp.sync.Lock._make(x[:20000], Fs, rp.sync.Detection(mod=lk.det.mod, baud=Fs/(sps*1.016), sps=sps*1.016,
                                                              beta=lk.det.beta, tau=0.0, M=lk.det.M, scores=lk.det.scores),
                             lk.points, lk.span, lk.det, lk.drop_db, lk.drift) # As if the first search were 1.6% off
    fixed = off.refine(x[:160000])
    assert abs(fixed.det.baud/(Fs/sps) - 1) < 1e-4 and fixed.found is fixed.det and fixed.stale is None
    assert fixed.quality.esn0_db > 15
