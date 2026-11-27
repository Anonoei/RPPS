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
    q = rp.sync.quality.measure(noisy(rng, QPSK, esn0), QPSK)
    assert q.esn0_db == pytest.approx(esn0, abs=0.3)
    assert q.ebn0_db == pytest.approx(esn0 - 10*np.log10(2), abs=0.3)

def test_bits_per_symbol_and_code_rate(rng):
    y = noisy(rng, PSK8, 12)
    q = rp.sync.quality.measure(y, PSK8)
    assert q.bits == 3 and q.ebn0_db == pytest.approx(q.esn0_db - 10*np.log10(3))
    q = rp.sync.quality.measure(y, PSK8, rate=0.5)
    assert q.ebn0_db == pytest.approx(q.esn0_db - 10*np.log10(1.5))

def test_snr_in_the_sample_bandwidth_inverts_awgn(rng):
    q = rp.sync.quality.measure(noisy(rng, QPSK, 10), QPSK, sps=4)
    assert q.snr_db == pytest.approx(q.esn0_db - 10*np.log10(4))
    assert rp.signal.channel.AWGN(esn0_db=10).snr_db(4) == pytest.approx(q.snr_db, abs=0.3)

def test_track_waits_then_follows(rng):
    t = rp.sync.quality.Track(QPSK, history=2000, minimum=500)
    assert t.update(noisy(rng, QPSK, 10, 100)) is None
    q = t.update(noisy(rng, QPSK, 10, 3000))
    assert q.n == 2000 and q.esn0_db == pytest.approx(10, abs=0.5)

def test_too_few_symbols():
    with pytest.raises(ValueError):
        rp.sync.quality.measure(QPSK[:2], QPSK)

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

def test_monitor_publishes_quality_and_passes_symbols(rng):
    y = noisy(rng, QPSK, 12, 4000)
    m = rp.sync.quality.Monitor(QPSK, minimum=500)
    d = rp.Data(y, sps=1)
    stored = d.data.copy()
    out = d + m
    assert np.array_equal(out.data, stored)
    assert abs(out.stream.est["quality"].esn0_db - 12) < 0.5

def test_monitor_in_a_profile():
    from rpps.profile import steps
    st = steps.build({"sync": "quality", "constellation": "psk/qpsk"}, rp.Rep.SYMBOLS, steps.context())
    (_, stage), = st.rx
    assert st.tx == [] and isinstance(stage, rp.sync.quality.Monitor) and stage.points is not None

# ---- errors and summary: what an analyser shows


def impaired(n, kI, kQ, c, sigma, seed=0):
    """QPSK symbols through kI I + kQ Q + c and noise (sigma per axis): a modulator's gain, skew and carrier leak"""
    rng = np.random.default_rng(seed)
    d = QPSK[rng.integers(0, 4, n)]
    y = kI*d.real + kQ*d.imag + c + sigma*(rng.standard_normal(n) + 1j*rng.standard_normal(n))
    return d, y

def test_summary_finds_offset_imbalance_and_skew():
    """The fit recovers what went in: 0.87 dB gain imbalance, 3 degrees of skew, a carrier leak 30 dB down"""
    kI, kQ = 1.05, 0.95*np.exp(1j*np.radians(93))
    c = 10**(-30/20)*np.sqrt((1.05**2 + 0.95**2)/2)
    _, y = impaired(20000, kI, kQ, c, 0.02)
    s = rp.sync.quality.summary(y, QPSK)
    assert s.gain_imbalance_db == pytest.approx(20*np.log10(1.05/0.95), abs=0.02)
    assert s.quad_error_deg == pytest.approx(3.0, abs=0.1)
    assert s.iq_offset_db == pytest.approx(-30.0, abs=0.3)
    assert s.n == 20000 and 0 <= s.evm_peak_at < 20000

def test_summary_of_clean_symbols_and_mer_is_the_snr():
    for esn0 in (15.0, 25.0, 35.0):
        sigma = np.sqrt(10**(-esn0/10)/2)
        d, y = impaired(20000, 1.0, 1j, 0.0, sigma, seed=int(esn0))
        s = rp.sync.quality.summary(y, QPSK)
        assert s.mer_db == pytest.approx(esn0, abs=0.3) # (no decision errors at these SNRs)
        assert s.evm_rms_pct == pytest.approx(100*10**(-esn0/20), rel=0.03)
        assert abs(s.gain_imbalance_db) < 0.05 and abs(s.quad_error_deg) < 0.3 and s.iq_offset_db < -25

def test_errors_per_symbol():
    d, y = impaired(1000, 1.0, 1j, 0.0, 0.0)
    y = y*1.1*np.exp(1j*np.radians(2)) # 10% large, turned 2 degrees
    E = rp.sync.quality.errors(y, QPSK)
    assert np.allclose(E.ideal, d) and np.allclose(E.vector, y - d)
    assert np.allclose(E.mag_pct, 10.0) and np.allclose(E.phase_deg, 2.0)
    assert np.allclose(E.evm_pct, 100*np.abs(1.1*np.exp(1j*np.radians(2)) - 1))

def test_summary_of_bpsk_has_no_imbalance():
    rng = np.random.default_rng(1)
    y = (2*rng.integers(0, 2, 5000) - 1) + 0.05*rng.standard_normal(5000)
    s = rp.sync.quality.summary(y.astype(complex), np.array([-1, 1], complex))
    assert s.gain_imbalance_db is None and s.quad_error_deg is None and s.mer_db > 20
