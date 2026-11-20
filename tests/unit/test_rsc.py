"""RSC constituents and a turbo interleaver, blind (rpps.analysis.code.conv.rsc)"""
import numpy as np

from rpps.analysis.code.conv import rsc

GF, GB = 0b1011, 0b1101 # 3GPP: feedforward 1 + D + D^3 (15 octal), feedback 1 + D^2 + D^3 (13 octal)

def test_3gpp_constituent():
    rng = np.random.default_rng(0)
    u = rng.integers(0, 2, 20000)
    p = rsc.encode(u, GF, GB)
    p = p ^ (rng.random(len(p)) < 0.02) # 2% errors
    c = rsc.constituent(u, p)
    assert (c.g_f, c.g_b) == (GF, GB) and (rsc.octal(c.g_f), rsc.octal(c.g_b)) == ("15", "13")

def test_feedforward_code_is_g_b_one():
    rng = np.random.default_rng(1)
    u = rng.integers(0, 2, 20000)
    p = rsc.encode(u, 0b10011, 1)
    c = rsc.constituent(u, p)
    assert c.g_b == 1 and c.g_f == 0b10011 and "feedforward" in str(c)

def test_independent_streams_have_none():
    rng = np.random.default_rng(2)
    assert rsc.constituent(rng.integers(0, 2, 20000), rng.integers(0, 2, 20000)) is None

def test_invert():
    u = np.random.default_rng(3).integers(0, 2, 500)
    assert np.array_equal(rsc.invert(rsc.encode(u, GF, GB), GF, GB), u)

def test_turbo_interleaver():
    rng = np.random.default_rng(4)
    K, F = 64, 40
    perm = rng.permutation(K)
    S = rng.integers(0, 2, (F, K))
    P1 = np.array([rsc.encode(s, GF, GB) for s in S])
    P2 = np.array([rsc.encode(s[perm], GF, GB) for s in S])
    assert rsc.constituent(S.reshape(-1), P1.reshape(-1)) is not None
    c2 = rsc.constituent(S.reshape(-1), P2.reshape(-1))
    assert c2 is None # the second parity has no check with s: interleaved
    got, agree = rsc.turbo_interleaver(S, P2, GF, GB)
    assert np.array_equal(got, perm) and np.all(agree == 1.0)

def test_turbo_interleaver_from_noisy_frames():
    """1024-bit frames at Es/N0 -1 dB (10% of the channel bits wrong), 300 frames: every position right, where the
    division by g_f (turbo_interleaver) gets a few percent"""
    rng = np.random.default_rng(5)
    K, F = 1024, 300
    perm = rng.permutation(K)
    S = rng.integers(0, 2, (F, K))
    P2 = np.array([rsc.encode(s[perm], GF, GB) for s in S])
    sigma = np.sqrt(1/(2*10**(-1/10)))
    llr = lambda b: 2*(1 - 2.0*b + sigma*rng.standard_normal(b.shape))/sigma**2
    got, conf = rsc.turbo_interleaver_soft(llr(S), llr(P2), GF, GB)
    assert np.array_equal(got, perm)
