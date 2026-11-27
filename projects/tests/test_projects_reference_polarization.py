"""projects/reference/polarization.py: which feed each carrier of a two-feed capture is on, its leakage, frequency reuse"""
import pathlib
import sys
from types import SimpleNamespace

import numpy as np
import pytest

PROJECTS = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECTS))
import _paths  # noqa: E402,F401
import polarization as P  # noqa: E402

Fs, N = 10e6, 2**19

def carrier(rng, mid, width, power):
    """Noise-like carrier: flat over `width` at `mid`, `power` over unit noise"""
    X = np.fft.fft(rng.standard_normal(N) + 1j*rng.standard_normal(N))
    X[np.abs(np.fft.fftfreq(N, 1/Fs) - mid) > width/2] = 0
    x = np.fft.ifft(X)
    return x*np.sqrt(power/np.mean(np.abs(x)**2))

def noise(rng):
    return (rng.standard_normal(N) + 1j*rng.standard_normal(N))*np.sqrt(0.5)

def at(pols, mid):
    return min(pols, key=lambda p: abs(p.sig.mid - mid))

@pytest.mark.parametrize("seed", range(4))
def test_leakage_reuse_and_unequal_gains(seed):
    rng = np.random.default_rng(seed)
    s1, s2, s3 = carrier(rng, -2e6, 1e6, 10.0), carrier(rng, 2e6, 1e6, 10.0), carrier(rng, 2e6, 1e6, 10.0)
    a = s1 + s2 + noise(rng)
    b = 3.0*(10**(-25/20)*np.exp(0.7j)*s1 + s3 + noise(rng)) # (the second feed's chain 9.5 dB hotter)
    pols = P.polarization(a, b, Fs=Fs, names=("H", "V"))
    assert len(pols) == 2
    leak, reuse = at(pols, -2e6), at(pols, 2e6)
    assert leak.name == "H" and not leak.xpd_bound and abs(leak.xpd - 25) < 2 and (leak.coherence is None or leak.coherence > 0.8)
    assert reuse.on == "reuse" and reuse.coherence < 0.1

@pytest.mark.parametrize("seed", range(4))
def test_deep_leakage_under_the_noise_and_a_slant(seed):
    rng = np.random.default_rng(seed)
    c1, c2 = carrier(rng, -3e6, 0.5e6, 3.0), carrier(rng, 1e6, 2e6, 3.0)
    a = c1 + c2 + noise(rng)
    b = 0.9*np.exp(2j)*c1 + 10**(-35/20)*c2 + noise(rng)
    pols = P.polarization(a, b, Fs=Fs)
    slant, deep = at(pols, -3e6), at(pols, 1e6)
    assert slant.on == "both" and slant.coherence > 0.9 and abs(slant.ratio) < 2
    assert deep.on == "a" and not deep.xpd_bound and abs(deep.xpd - 35) < 2.5 and not deep.leak_seen

def test_a_carrier_on_one_feed_found_by_the_others_survey():
    rng = np.random.default_rng(7)
    a, b = noise(rng), carrier(rng, 1e6, 1e6, 3.0) + noise(rng)
    (p,) = P.polarization(a, b, Fs=Fs, names=("RHCP", "LHCP"))
    assert p.name == "LHCP" and p.xpd_bound and p.xpd > 40 and p.cn[0] is None

def test_a_leak_over_the_noise_is_the_strong_feeds():
    rng = np.random.default_rng(3)
    s = carrier(rng, 0.0, 1e6, 300.0)
    (p,) = P.polarization(s + noise(rng), 10**(-12/20)*s + noise(rng), Fs=Fs, names=("H", "V"))
    assert p.name == "H" and p.leak_seen and abs(p.xpd - 12) < 1 and "its leakage" in str(p)

class Src:
    def __init__(self, x, Fs, cf, pol=None, freq=None):
        self.x, self.Fs, self.cf = x, Fs, cf
        self.context = SimpleNamespace(freq=cf if freq is None else freq, chain=SimpleNamespace(polarization=pol))
    def peek(self, n):
        return self.x[:n]

def test_feeds_named_by_their_chains_and_checked_alike():
    rng = np.random.default_rng(1)
    s = carrier(rng, 0.0, 1e6, 10.0)
    pols = P.polarization(Src(s + noise(rng), Fs, 1e9, "V"), Src(noise(rng), Fs, 1e9, "H"))
    assert pols and pols[0].name == "V"
    with pytest.raises(ValueError, match="sample rates"):
        P.polarization(Src(s, Fs, 1e9), Src(s, 2*Fs, 1e9))
    with pytest.raises(ValueError, match="centre frequencies"):
        P.polarization(Src(s, Fs, 1e9), Src(s, Fs, 1.001e9))
    with pytest.raises(ValueError, match="contexts"):
        P.polarization(Src(s, Fs, 1e9), Src(s, Fs, 1e9, freq=2e9))

def test_neighbours_edge_to_edge_on_opposite_feeds():
    """Carriers whose bands touch don't spill into each other's measurement (the inner part of each band)"""
    rng = np.random.default_rng(5)
    h, v = carrier(rng, 1e6, 1e6, 300.0), carrier(rng, 2e6, 1e6, 300.0)
    pols = P.polarization(h + noise(rng), v + noise(rng), Fs=Fs, names=("H", "V"))
    assert [(p.name, p.xpd_bound) for p in pols] == [("H", True), ("V", True)]
