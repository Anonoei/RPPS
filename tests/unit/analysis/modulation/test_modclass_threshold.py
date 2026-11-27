"""Modulation detection thresholds: each detector names each constellation above the Es/N0 tools/modclass_bench.py
measured (Pc >= 0.85 at 2 dB above it), and not far below it (Pc < 0.9 at 4 dB below: the number is a threshold)"""
import numpy as np
import pytest

import rpps as rp
from rpps import symbol
from rpps.analysis.modulation import cumulants as C

from support.signals import shaped

# tools/modclass_bench.py (2026-11-23, detect scoring at the in-band Es/N0), 100 trials, Pc >= 0.9 up to 20 dB
# (cumulants: 4096 symbols; detect: 2048 at 4 sps, rate searched). BLIND QAM and ASK: the family. Left out, as their Pc
# hovers about 0.9 rather than crossing it: BLIND 4ASK (read as BPSK now and then at any SNR), BLIND 64QAM (dips under
# 0.9 at 2-4 dB, read as 32APSK); AIDED 64QAM and 256QAM never reach 0.9 at 4096 symbols (test_cumulants gives them
# 20000 and 200000). ESTIMATED: Es/N0 from rp.sync.quality.cyclic on the samples (BPSK left out: its -6.25 is under
# where the estimate works, about -3 dB). DETECT: each candidate at its own M2M4 Es/N0 (the fit above 12.5 dB).
# HOVERS: Pc at or about 0.9 well under the bench's threshold, no crossing to check below it (aided BPSK, at -10 dB
# still; detect 8PSK, 16PSK its rival)
BLIND = {"BPSK": 6.5, "QPSK": 6.75, "8PSK": 7.0, "16PSK": 7.5, "16QAM": -0.5, "32QAM": 6.5, "256QAM": 0.0,
         "16APSK": 9.25, "32APSK": 8.75, "8ASK": 6.25}
AIDED = {"BPSK": -2.5, "QPSK": -0.25, "8PSK": 1.25, "16PSK": 7.5, "16QAM": 6.75, "32QAM": 3.5, "16APSK": 6.25,
         "32APSK": 4.5, "4ASK": 5.0, "8ASK": 4.5} # 4ASK: two bench draws said 5.0 and 3.75 (other seed sets 0.85 at 5.75)
ESTIMATED = {"QPSK": 0.75, "8PSK": 1.25, "16PSK": 7.75, "16QAM": 5.75, "32QAM": 3.75, "16APSK": 5.75,
             "32APSK": 5.0, "4ASK": 3.5, "8ASK": 8.0}
DETECT = {"BPSK": -6.75, "QPSK": 4.5, "8PSK": 9.0, "16PSK": 8.25, "16QAM": 10.0, "32QAM": 5.5, "64QAM": 10.0,
          "16APSK": 7.25, "32APSK": 8.0}
NAMES = ("BPSK", "QPSK", "8PSK", "16PSK", "16QAM", "32QAM", "64QAM", "16APSK", "32APSK")
HOVERS = {("aided", "BPSK"), ("detect", "8PSK")}
# The bench's threshold is where Pc first holds 0.9; 60 trials scatter about +-0.04 there and some curves climb 4% a
# dB (aided 16QAM, 4ASK), so: Pc >= 0.85 at 2 dB above, Pc < 0.9 at 4 dB below (over 5 other seed sets the lowest
# above was 0.88, blind 16QAM; the highest below 0.83, aided 16QAM)
TRIALS, ABOVE, BELOW, FLOOR = 60, 2.0, 4.0, 0.85
SEED = 100

def family(name):
    return next((f for f in ("QAM", "ASK") if name.endswith(f)), name)

def noisy(rng, pts, esn0, n=4096):
    pts = np.asarray(pts)/np.sqrt(np.mean(np.abs(pts)**2))
    s = pts[rng.integers(0, len(pts), n)]*np.exp(2j*np.pi*rng.random())
    return s + 10**(-esn0/20)*(rng.standard_normal(n) + 1j*rng.standard_normal(n))/np.sqrt(2)

def pc(named, truth, esn0, seed):
    return np.mean([named(np.random.default_rng(seed + i), esn0) == truth for i in range(TRIALS)])

def cumulant(name, aided):
    pts = C.candidates()[name]
    def named(rng, esn0):
        got = C.classify(noisy(rng, pts, esn0), esn0_db=esn0 if aided else None)[0].name
        return got if aided else family(got)
    return named, name if aided else family(name)

def estimated(name):
    """Symbols from RRC samples (4 sps) at the known timing, Es/N0 from the samples"""
    from rpps.sync import acquire as A
    pts = C.candidates()[name]
    def named(rng, esn0):
        tau = rng.random()
        x, _ = shaped(rng, pts, 4096, 4, esn0_db=esn0, tau=tau)
        x = x*np.exp(2j*np.pi*rng.random())
        c = A.centres_in(len(x), tau*4 + np.ceil((12*4 + 5 - tau*4)/4)*4, 4, 12)
        est = rp.sync.quality.cyclic(x, 4)
        es = est.esn0_db if est is not None and np.isfinite(est.esn0_db) else None
        return C.classify(A.matrix_mf(x, c, 4, 0.35, 12), esn0_db=es)[0].name
    return named, name

def detector(name):
    pts = symbol.load_or_generate(name).points
    def named(rng, esn0):
        x, _ = shaped(rng, pts, 2048, 4, esn0_db=esn0, tau=rng.random(), cfo=rng.uniform(-1e-3, 1e-3))
        try:
            return rp.sync.detect(x, 1e6, names=NAMES).mod
        except rp.errors.RPPSError:
            return None
    return named, name

def check(named, truth, thr, below=True):
    assert pc(named, truth, thr + ABOVE, SEED) >= FLOOR
    if below and thr - BELOW > -10:
        assert pc(named, truth, thr - BELOW, SEED + TRIALS) < 0.9

@pytest.mark.parametrize("name", sorted(BLIND))
def test_cumulants_blind(name):
    check(*cumulant(name, False), BLIND[name])

@pytest.mark.parametrize("name", sorted(AIDED))
def test_cumulants_aided(name):
    check(*cumulant(name, True), AIDED[name], below=("aided", name) not in HOVERS)

@pytest.mark.slow
@pytest.mark.parametrize("name", sorted(ESTIMATED))
def test_cumulants_estimated(name):
    check(*estimated(name), ESTIMATED[name])

@pytest.mark.slow
@pytest.mark.parametrize("name", sorted(DETECT))
def test_detect(name):
    check(*detector(name), DETECT[name], below=("detect", name) not in HOVERS)
