"""projects/reference/power.py: a receiver's level calibration (dBFS to dBm at the port) from its floor, a tone, a noise
source"""
import math
import pathlib
import sys

import numpy as np
import pytest

PROJECTS = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECTS))
import _paths  # noqa: E402,F401
import power as W  # noqa: E402

Fs, N = 10e6, 2**19

def receiver(rng, gain_db, nf_db, extra_dbm_hz=None, tone_dbm=None):
    """Samples a receiver makes: dBm at the port becomes dBFS + gain_db (gain_db: the offset negated), its own noise
    kT0 + NF, a noise source's on top, a tone at 1 MHz"""
    n0 = W.KT0 + nf_db if extra_dbm_hz is None else 10*math.log10(10**((W.KT0 + nf_db)/10) + 10**(extra_dbm_hz/10))
    p = 10**((n0 + gain_db)/10)*Fs # (power over the band: density times Fs)
    x = (rng.standard_normal(N) + 1j*rng.standard_normal(N))*math.sqrt(p/2)
    if tone_dbm is not None:
        x = x + math.sqrt(10**((tone_dbm + gain_db)/10))*np.exp(2j*np.pi*1e6/Fs*np.arange(N))
    return x

@pytest.mark.parametrize("seed", range(3))
def test_the_floor_with_the_noise_figure_gives_the_offset(seed):
    rng = np.random.default_rng(seed)
    x = receiver(rng, 100.0, 6.0) # (-168 dBm/Hz at the port is -68 dBFS/Hz)
    fl = W.floor_dbfs_hz(x, Fs)
    assert fl == pytest.approx(-68.0, abs=0.1)
    lv = W.floor(fl, 6.0)
    assert lv.offset == pytest.approx(-100.0, abs=0.1) and lv.accuracy == pytest.approx(math.hypot(2.0, 0.2))

@pytest.mark.parametrize("seed", range(3))
def test_a_tone_of_known_power(seed):
    rng = np.random.default_rng(seed)
    x = receiver(rng, 100.0, 6.0, tone_dbm=-70.0)
    lv = W.tone(W.tone_dbfs(x, Fs, 1e6), -70.0)
    assert lv.offset == pytest.approx(-100.0, abs=0.2)
    with pytest.raises(ValueError, match="no tone"):
        W.tone_dbfs(x, Fs, -3e6, span=100e3)

@pytest.mark.parametrize("seed", range(3))
def test_a_noise_source_measures_the_noise_figure(seed):
    rng = np.random.default_rng(seed)
    enr = 15.0
    on = receiver(rng, 100.0, 5.0, extra_dbm_hz=W.KT0 + enr)
    off = receiver(rng, 100.0, 5.0)
    lv = W.y_factor(W.floor_dbfs_hz(on, Fs), W.floor_dbfs_hz(off, Fs), enr)
    assert lv.nf == pytest.approx(5.0, abs=0.15) and lv.offset == pytest.approx(-100.0, abs=0.2) and lv.accuracy < 0.4
    with pytest.raises(ValueError, match="isn't reaching"):
        W.y_factor(-70.0, -70.0, enr)
