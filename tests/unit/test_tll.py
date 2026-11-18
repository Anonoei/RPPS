"""The timing locked loop and the timing error detectors it is built from (rpps.sync.time.ed, rpps.sync.pll)"""
import numpy as np
import pytest

import rpps as rp
from rpps.sync import pll
from rpps.sync.time import ed

P = np.exp(1j*np.pi/4*(2*np.arange(4) + 1))

def matched_qpsk(n=6000, sps=4, delay=0.0, seed=0):
    """QPSK through RRC and its matched filter (a raised cosine), delayed by a fraction of a sample"""
    rng = np.random.default_rng(seed)
    syms = P[rng.integers(0, 4, n)]
    h = rp.signal.filters.ShapingRRC(None, sps, 0.35).pulse
    up = np.zeros(n*sps, complex)
    up[::sps] = syms
    x = np.convolve(np.convolve(up, h), h)/np.sum(h*h)
    t = np.arange(len(x) - 10) + delay
    return (np.interp(t, np.arange(len(x)), x.real) + 1j*np.interp(t, np.arange(len(x)), x.imag)).astype(np.complex64)

def evm(y):
    y = y[-2000:]/np.sqrt(np.mean(np.abs(y[-2000:])**2))
    return float(np.sqrt(np.mean(np.min(np.abs(y[:, None] - P[None, :])**2, axis=1))))

@pytest.mark.parametrize("delay", [0.0, 1.3, 2.7])
def test_tll_locks_like_oerder_meyr(delay):
    x = matched_qpsk(delay=delay)
    tll = evm((rp.Data(x, Fs=4e3, sps=4) + rp.sync.time.TLL(4, bw=0.01)).data)
    om = evm((rp.Data(x, Fs=4e3, sps=4, flush=True) + rp.sync.time.OerderMeyr(4)).data)
    assert tll < om + 0.02 # (both at the cubic interpolation's floor at 4 samples per symbol)

def test_tll_any_block_cut():
    x = matched_qpsk(2000)
    def run(cut):
        stage, st, out, i = rp.sync.time.TLL(4), None, [], 0
        for b in cut:
            d = rp.Data(x[i:i+b], stream=st, Fs=4e3, sps=4)
            st = d.stream
            y = d + stage
            out.append(y.data)
            i += b
        assert y.sps == 1 and y.Fs == 1e3
        return np.concatenate(out)
    whole = run([len(x)])
    np.testing.assert_allclose(run([1, 0, 777, 3000, len(x) - 3778]), whole, atol=1e-6)

def test_error_detectors():
    assert np.real(ed.zx.zx_nd(1 + 0j, 0.2 + 0j, -1 + 0j)) == pytest.approx(0.4)
    for cls in (ed.ZX_ND, ed.EL_ND):
        d = cls()
        for v in (1 + 1j, 0.5 - 0.2j, -1 - 1j):
            d.input(v)
        assert np.isfinite(d.err)

def test_loop_gain_helpers_agree():
    a, b = pll.loop_gains(0.01)
    loop = pll.PILoop.from_bandwidth(0.01)
    assert (loop.Kp, loop.Ki) == (a, b)
    assert loop.run(1.0) == pytest.approx(a + b)
    kp, ki = pll.K_fs(1.0, 1.0, 0.7071, 100.0, 1e4)
    kp2, ki2 = pll.K_sr(1.0, 1.0, 0.7071, 100.0, 1e4, 1)
    assert (kp, ki) == pytest.approx((kp2, ki2)) and kp > 0 and ki > 0
