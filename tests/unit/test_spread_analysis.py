"""DSSS under the noise: code period, chip rate, code, despreading (rpps.analysis.sequence.spread)"""
import numpy as np
import pytest

from rpps.analysis.sequence import spread
from conftest import shaped

def mseq(poly, L):
    st, out = [1]*L, []
    taps = [j for j in range(1, L + 1) if poly >> j & 1]
    for _ in range(2**L - 1):
        out.append(st[-1])
        v = 0
        for j in taps:
            v ^= st[j - 1]
        st = [v] + st[:-1]
    return 1 - 2*np.array(out)

C127 = mseq(0b10001001, 7)

def dsss(rng, code, nsym, snr_db, sps=4.0):
    d = 1 - 2*rng.integers(0, 2, nsym)
    chips = (d[:, None]*code[None, :]).reshape(-1)
    x, _ = shaped(rng, [1], len(chips), sps, beta=0.35, snr_db=snr_db, sym=chips.astype(complex), tau=0.37)
    return x, d

def match(c, code):
    P = len(code)
    return max(abs(np.dot(np.roll(c, j), code))/P for j in range(P)) if len(c) == P else 0.0

@pytest.mark.parametrize("snr", [-5.0, -10.0])
def test_period_rate_code_and_symbols(snr):
    rng = np.random.default_rng(0)
    x, d = dsss(rng, C127, 1200, snr)
    p = spread.period(x, 1e6, max_period=800)
    assert p.significant and abs(p.lag - 508) <= 1
    R, k, T = spread.chip_rate(x, 1e6, p.lag, fluct=p.fluct)
    assert k == 127 and R == pytest.approx(250e3, rel=1e-3)
    c = spread.code(x, 1e6, R, k)
    assert match(c.code, C127) > 0.97
    y = spread.despread(c.chips_rx, c.code)
    assert np.mean(np.abs(y.real)) > 3*np.std(np.abs(y.real)) # two clean clusters: the processing gain

def test_detection_goes_lower_than_recovery():
    """A 127-chip code over 600k samples: the period is found at -13 dB, where the chip rate's |x|^2 line is gone (code
    recovery needs that line: down to about -11 dB here, where the narrow-window search still finds it)"""
    rng = np.random.default_rng(0)
    x, _ = dsss(rng, C127, 1200, -13.0)
    p = spread.period(x, 1e6, max_period=800)
    assert p.significant and abs(p.fine - 508) < 0.5
    rng = np.random.default_rng(0)
    x, _ = dsss(rng, C127, 1200, -11.0)
    p = spread.period(x, 1e6, max_period=800)
    R, k, T = spread.chip_rate(x, 1e6, p.lag, fluct=p.fluct, fine=p.fine)
    assert k == 127 and match(spread.code(x, 1e6, R, k).code, C127) > 0.95

def test_noise_has_no_period():
    rng = np.random.default_rng(1)
    g = rng.standard_normal(300000) + 1j*rng.standard_normal(300000)
    assert not spread.period(g, 1e6, max_period=800).significant
