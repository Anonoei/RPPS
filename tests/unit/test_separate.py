"""Two co-channel signals of equal power separated (rpps.analysis.spectrum.separate)"""
import numpy as np
import pytest

import rpps as rp
from rpps.signal.filters import rrc_pulse
from rpps.analysis.spectrum import separate as SP

def shaped(sym, sps, tau, beta=0.35, span=8):
    n = int((len(sym) + 2*span + 2)*sps)
    out = np.zeros(n, complex)
    L = int(span*sps)
    for i, s in enumerate(sym):
        c = (i + span + tau)*sps
        a = int(c) - L
        out[a:a + 2*L + 1] += s*rrc_pulse((np.arange(a, a + 2*L + 1) - c)/sps, beta)
    return out

def pair(rng, f2, n=6000, fs=200e3, baud=50e3):
    q = rp.symbol.load("qpsk").points
    s1, s2 = q[rng.integers(0, 4, n)], q[rng.integers(0, 4, n)]
    x1, x2 = shaped(s1, 4, 0.0), shaped(s2, 4, 0.37)
    t = np.arange(len(x1))/fs
    x = x1*np.exp(1j*0.4 + 2j*np.pi*300*t) + x2*np.exp(1j*2.0 + 2j*np.pi*f2*t)
    p = np.mean(np.abs(x1)**2)
    x += np.sqrt(p*10**(-15/10)/2)*(rng.standard_normal(len(x)) + 1j*rng.standard_normal(len(x)))
    return x, fs, baud, s1, s2, q

def ser(dec, s):
    best = 1.0
    for sh in range(-3, 4):
        a, b = dec[max(0, sh):], s[max(0, -sh):]
        n = min(len(a), len(b))
        best = min(best, min(np.mean(np.abs(a[:n] - b[:n]*1j**k) > 1e-6) for k in range(4)))
    return best

@pytest.mark.parametrize("f2", [-450.0, 280.0])
def test_equal_power_pair_is_separated(f2):
    x, fs, baud, s1, s2, q = pair(np.random.default_rng(0), f2)
    ps = SP.estimate(x, fs, baud)
    lo, hi = sorted([300.0, f2])
    assert abs(ps[0].cfo - lo) < 2 and abs(ps[1].cfo - hi) < 2
    r = SP.separate(x, fs, baud, params=ps)
    first, second = (s2, s1) if f2 < 300 else (s1, s2)
    assert ser(r.symbols[0], first) == 0 and ser(r.symbols[1], second) == 0
    assert max(r.evm) < 0.15
