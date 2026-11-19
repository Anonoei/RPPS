import numpy as np
import pytest

import rpps as rp

QPSK = np.exp(1j*np.pi/4*np.array([1, 3, 5, 7]))
PSK8 = np.exp(2j*np.pi*np.arange(8)/8)

def noisy(rng, pts, esn0_db, n=30000, gain=0.37):
    sym = pts[rng.integers(0, len(pts), n)]
    noise = (rng.standard_normal(n) + 1j*rng.standard_normal(n))/np.sqrt(2)*10**(-esn0_db/20)
    return gain*(sym + noise) # An arbitrary gain: the estimate must not depend on it

@pytest.mark.parametrize("esn0", [6, 10, 14])
def test_measure_recovers_esn0_and_ebn0(rng, esn0):
    q = rp.analysis.modulation.quality.measure(noisy(rng, QPSK, esn0), QPSK)
    assert q.esn0_db == pytest.approx(esn0, abs=0.3)
    assert q.ebn0_db == pytest.approx(esn0 - 10*np.log10(2), abs=0.3)

def test_bits_per_symbol_and_code_rate(rng):
    y = noisy(rng, PSK8, 12)
    q = rp.analysis.modulation.quality.measure(y, PSK8)
    assert q.bits == 3 and q.ebn0_db == pytest.approx(q.esn0_db - 10*np.log10(3))
    q = rp.analysis.modulation.quality.measure(y, PSK8, rate=0.5)
    assert q.ebn0_db == pytest.approx(q.esn0_db - 10*np.log10(1.5))

def test_snr_in_the_sample_bandwidth_inverts_awgn(rng):
    q = rp.analysis.modulation.quality.measure(noisy(rng, QPSK, 10), QPSK, sps=4)
    assert q.snr_db == pytest.approx(q.esn0_db - 10*np.log10(4))
    assert rp.signal.channel.AWGN(esn0_db=10).snr_db(4) == pytest.approx(q.snr_db, abs=0.3)

def test_track_waits_then_follows(rng):
    t = rp.analysis.modulation.quality.Track(QPSK, history=2000, minimum=500)
    assert t.update(noisy(rng, QPSK, 10, 100)) is None
    q = t.update(noisy(rng, QPSK, 10, 3000))
    assert q.n == 2000 and q.esn0_db == pytest.approx(10, abs=0.5)

def test_too_few_symbols():
    with pytest.raises(ValueError):
        rp.analysis.modulation.quality.measure(QPSK[:2], QPSK)

def test_gate_passes_clean_symbols(rng):
    g = rp.analysis.modulation.quality.gate(noisy(rng, PSK8, 14, 20000), PSK8)
    assert g.ok and abs(g.rotation) < 1 and g.near < 2*g.near_expected

def test_gate_catches_a_turned_payload(rng):
    """Frames turned ~30 deg by a bad phase reference read as noise to a rank test"""
    y = noisy(rng, QPSK, 15, 20000)*np.exp(1j*np.radians(30))
    g = rp.analysis.modulation.quality.gate(y, QPSK)
    assert not g.ok and g.rotation == pytest.approx(30, abs=3) or "near a boundary" in g.why

def test_gate_catches_a_small_turn_for_8psk(rng):
    g = rp.analysis.modulation.quality.gate(noisy(rng, PSK8, 16, 20000)*np.exp(1j*np.radians(6)), PSK8)
    assert not g.ok and "turned" in g.why

def test_gate_catches_drift(rng):
    y = noisy(rng, QPSK, 15, 20000)*np.exp(1j*np.linspace(-0.3, 0.3, 20000)) # a residual frequency: +-17 deg
    g = rp.analysis.modulation.quality.gate(y, QPSK)
    assert not g.ok and "drifting" in g.why

@pytest.mark.parametrize("k, ok", [(0.15, True), (0.6, False)])
def test_gate_measures_compression(rng, k, ok):
    """16QAM through a compressing amplifier: the clusters leave their points; mild compression is reported, strong
    compression (outer points pulled most of the way to the next ring) fails the gate"""
    pts = rp.symbol.QAM(16).points
    pts = np.asarray(pts)/np.sqrt(np.mean(np.abs(pts)**2))
    y = noisy(rng, pts, 22, 20000, gain=1.0)
    y = y/(1 + k*np.abs(y)**2)
    g = rp.analysis.modulation.quality.gate(y, pts)
    assert g.displaced > 0.05 and g.ok == ok
