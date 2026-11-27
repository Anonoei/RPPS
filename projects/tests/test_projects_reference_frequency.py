"""projects/reference/frequency.py: the receive chain's own errors (the radio's reference, an LNB's LO), from known
carriers, known rates and LTE cells"""
import pathlib
import sys

import numpy as np
import pytest

import rpps as rp

PROJECTS = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECTS))
import _paths  # noqa: E402,F401
import frequency as R  # noqa: E402

class Chain: # (what fit reads of a pysdrlib caps.Chain: rpps doesn't need pysdrlib for this)
    def __init__(self, lo, invert):
        self.lo, self.invert = lo, invert
    def to_if(self, rf):
        return self.lo - rf if self.invert else rf - self.lo

class Ctx:
    def __init__(self, chain=None, clock_source=None):
        self.chain, self.clock_source = chain, clock_source

def seen(true, ppm, lo_error=0.0, chain=None):
    """Where a receiver with these errors puts a carrier at `true` (the model fit inverts)"""
    if chain is None:
        return true - true*ppm*1e-6
    s = -1.0 if chain.invert else 1.0
    return true - lo_error - s*chain.to_if(true)*ppm*1e-6

def test_a_known_carrier_gives_the_radios_ppm():
    refs = [R.FrequencyReference(seen(751e6, 15.0), true=751e6, sigma=50.0)]
    cal = R.fit(refs)
    assert cal.ppm == pytest.approx(15.0, abs=0.1) and cal.ppm_sigma < 0.1
    f, sd = cal.frequency(seen(1965.85e6, 15.0)) # another carrier of the same capture
    assert f == pytest.approx(1965.85e6, abs=3*sd + 1) and sd < 200
    assert cal.rate(10.762238e6/(1 + 15e-6))[0] == pytest.approx(10.762238e6, rel=1e-8) # a fast clock reads rates low

@pytest.mark.parametrize("invert", [False, True])
def test_an_lnb_and_the_radio_come_apart(invert):
    """Through an LNB (its LO below the signal, Ku; above, C-band), carriers at two IFs and a known rate give both"""
    chain = Chain(9.75e9 if not invert else 5.15e9, invert)
    ctx = Ctx(chain)
    ppm, lo = -12.0, 35e3
    if invert:
        truths = [3.9e9, 4.1e9]
    else:
        truths = [10.95e9, 11.70e9]
    refs = [R.FrequencyReference(seen(t, ppm, lo, chain), true=t, sigma=30.0) for t in truths]
    refs.append(R.rate(23.8e6/(1 + ppm*1e-6), 23.8e6, sigma_ppm=0.2))
    cal = R.fit(refs, ctx)
    assert cal.ppm == pytest.approx(ppm, abs=0.1) and cal.lo_error == pytest.approx(lo, abs=200)
    f, sd = cal.frequency(seen(truths[0] + 20e6, ppm, lo, chain))
    assert f == pytest.approx(truths[0] + 20e6, abs=3*sd + 1)

def test_a_raster_is_resolved_only_when_the_rest_pins_it():
    """An LTE carrier alone at 751 MHz: 25 ppm allows 19 kHz, under a quarter of the 100 kHz raster, so it is; at 3.5
    GHz, 88 kHz is not, so the raster points it could be are listed"""
    low = R.fit([R.FrequencyReference(seen(751e6, 15.0), raster=R.LTE_RASTER)])
    assert not low.ambiguous and low.ppm == pytest.approx(15.0, abs=0.1)
    high = R.fit([R.FrequencyReference(seen(3.55e9, 15.0), raster=R.LTE_RASTER)])
    (ref, could), = high.ambiguous
    assert 3.55e9 in could and len(could) > 1 and high.ppm == 0.0 # (nothing measured: the prior)
    with_rate = R.fit([R.FrequencyReference(seen(3.55e9, 15.0), raster=R.LTE_RASTER), R.RateReference(14.6, 0.5)])
    assert not with_rate.ambiguous and with_rate.ppm == pytest.approx(15.0, abs=0.05)

def test_a_disciplined_reference_is_trusted():
    assert R.priors(Ctx(clock_source="gpsdo"))[1] == R.DISCIPLINED_PPM and R.priors(None) == (0.0, R.RADIO_PPM)
    assert R.priors(Ctx(Chain(9.75e9, False)))[0] == R.LNB_HZ

def stretched_lte(ppm, cf, fs=1.92e6, frames=50, seed=1):
    """An LTE cell at cf exactly, received by a radio whose clock runs ppm fast: tuned cf (1 + ppm), sampled fs (1 + ppm)"""
    from rpps.grid import lte
    from rpps.math.interp import interpolate
    x = lte.tx.channel(lte.tx.downlink(pci=179, ports=1, n_rb=6, frames=frames), fs=fs, delay=3000, snr_db=10, seed=seed)
    k = np.arange(int(len(x)*(1 + ppm*1e-6)) - 32)
    y = interpolate(x.astype(np.complex128), k/(1 + ppm*1e-6))
    return (y*np.exp(-2j*np.pi*cf*ppm*1e-6*k/(fs*(1 + ppm*1e-6)))).astype(np.complex64)

def test_an_lte_cell_calibrates_the_capture():
    y = stretched_lte(15.0, 751e6)
    freq, clock = R.lte(y, 1.92e6, cf=751e6)
    assert freq.raster == R.LTE_RASTER and freq.measured == pytest.approx(751e6 - 751e6*15e-6, abs=100)
    assert clock.ppm == pytest.approx(15.0, abs=1.5)
    cal = R.calibrate(rp.source.Memory(y, Fs=1.92e6, cf=751e6))
    assert cal.ppm == pytest.approx(15.0, abs=0.2) and len(cal.used) == 2 and not cal.ambiguous
    assert R.lte(np.zeros(200000, np.complex64), 1.92e6) == [] # (no cell: no references)

def test_extents_join_an_ofdm_carriers_pieces():
    class Box:
        def __init__(self, mid, width, snr=20.0):
            self.mid, self.width, self.snr = mid, width, snr
    pieces = [(-4.55e6, -4.1e6), (-4.05e6, -2.5e6), (-2.45e6, -0.3e6), (-0.25e6, 2.25e6), (2.3e6, 4.5e6)] # (gaps of 50 kHz)
    boxes = [Box((a + b)/2, b - a) for a, b in pieces] + [Box(9e6, 1e6), Box(7e6, 50e3, snr=3.0)] # (another, and noise)
    assert [tuple(round(v/1e3) for v in e) for e in R.extents(boxes)] == [(-4550, 4500), (8500, 9500)]
