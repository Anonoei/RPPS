"""802.11b through its profiles, with the PLCP rate announced late"""
import numpy as np
import pytest

import rpps as rp
from rpps.base.types import Rep
from rpps.profile import runner as runner_mod
from support.signals import hard
from support.dvbs2 import KNOWN, seeded_bits as bits

@pytest.mark.parametrize("rate,noise", [("1m", 0.4), ("2m", 0.28)]) # DQPSK needs about 3 dB more than DBPSK
def test_80211b_loopback_through_noise(rate, noise):
    rng = np.random.default_rng(1)
    p = rp.profile.load(f"ieee80211b/{rate}", sync=KNOWN)
    msg = rng.integers(0, 256, 40).astype(np.uint8)
    wave = p.tx()(rp.Data(msg, Fs=1e6, sps=1))
    noisy = rp.Data(wave.data + noise*(rng.normal(size=len(wave)) + 1j*rng.normal(size=len(wave))).astype(np.complex64), Fs=wave.Fs, sps=wave.sps, delay=wave.delay)
    out = p.rx()(noisy)
    assert out.REP == Rep.DATA
    got, bits = hard(out), np.unpackbits(msg).astype(bool)
    assert len(got) >= len(bits) - 32
    # Past the differential start and the descrambler's first 7 bits; each channel error is 3 after the descrambler
    assert np.mean(got[16:] != bits[16:len(got)]) < 0.01

# ---- 802.11b: the PLCP rate, announced late

def b_transmit(packets, rates):
    p = rp.profile.load("ieee80211b/long", sync=KNOWN)
    tx, s, waves = p.tx(), None, []
    for i, (psdu, r) in enumerate(zip(packets, rates)):
        d = rp.Data(psdu, Fs=1e6, sps=1, stream=s, flush=i == len(packets) - 1)
        s = d.stream
        o = tx(d, fields={"signal": r})
        waves.append(o.data)
    return p, np.concatenate(waves + [np.zeros(400, np.complex64)]), o

def b_receive(p, x, o, cuts=None, noise=0.0):
    if noise:
        x = (x + noise*(np.random.default_rng(9).normal(size=len(x)) + 1j*np.random.default_rng(10).normal(size=len(x)))).astype(np.complex64)
    rx, st, outs, i = p.rx(), None, [], 0
    cuts = cuts or [len(x)]
    for k, n in enumerate(cuts):
        d = rp.Data(x[i:i + n], Fs=o.Fs, sps=4, delay=o.delay, stream=st, flush=k == len(cuts) - 1)
        st, i = d.stream, i + n
        outs.append(np.asarray(rx(d).data).astype(bool))
    return np.concatenate(outs), st

@pytest.fixture(scope="module")
def b_signal():
    packets = [bits(8*n, i) for i, n in enumerate([40, 25, 60, 33])]
    rates = [20, 10, 20, 20]
    p, x, o = b_transmit(packets, rates)
    return p, x, o, packets, rates

def test_80211b_rate_switches_mid_packet(b_signal):
    p, x, o, packets, rates = b_signal
    got, st = b_receive(p, x, o)
    ref = np.concatenate(packets)
    assert len(got) == len(ref)
    assert np.array_equal(got[:-2], ref[:-2]) # (the end of the stream cuts the last symbol's pulse tail)
    ctl = st.control
    heads = [f for _, f, _ in ctl.fields("plcp") if "signal" in f]
    assert [f["signal"] for f in heads] == rates and [f["bits"] for f in heads] == [len(b) for b in packets]
    assert ctl.rerun == len(packets) and not ctl.reports # each header re-ran its payload, nothing dropped
    assert st.est["ieee80211b_plcp"]["frames"] == 4 and st.est["ieee80211b_plcp"]["crc_failed"] == 0

def test_80211b_any_block_cut(b_signal):
    p, x, o, packets, rates = b_signal
    ref, _ = b_receive(p, x, o)
    for cuts in ([1000]*(len(x)//1000) + [len(x) % 1000], [37]*(len(x)//37) + [len(x) % 37], [5000, 3, 20000, len(x) - 25003]):
        got, st = b_receive(p, x, o, cuts)
        assert np.array_equal(got, ref), cuts[:3]
        assert not st.control.reports

def test_80211b_through_noise(b_signal):
    p, x, o, packets, rates = b_signal
    got, st = b_receive(p, x, o, noise=0.3)
    ref = np.concatenate(packets)
    assert len(got) == len(ref) and np.mean(got != ref) < 0.01 # each channel error is three after the descrambler

def test_too_late_is_dropped_and_reported(b_signal, monkeypatch):
    """A buffer too short for the header's reach: the change is applied from the oldest position still held, and the drop
    reported (which change, from where, how much was lost)"""
    p, x, o, packets, rates = b_signal
    monkeypatch.setattr(runner_mod, "MIN_SEGMENT", 4)
    p = rp.profile.load("ieee80211b/long", sync=KNOWN)
    p.controls["plcp"].reach = 1
    got, st = b_receive(p, x[:len(x)//3], o)
    drops = [r for r in st.control.reports if r["what"] == "dropped"]
    assert drops and drops[0]["step"] == "plcp" and drops[0]["lost"] > 0 and drops[0]["applied"] > drops[0]["at"] - 1000
    assert st.est["ieee80211b_plcp"]["frames"] >= 1 # and it went on

def test_80211b_transmit_fields_from_the_schedule():
    p = rp.profile.load("ieee80211b/long", signal=10)
    o = p.tx()(rp.Data(bits(80, 1), Fs=1e6, sps=1))
    assert [f["signal"] for _, f, _ in o.stream.control.fields("plcp") if "bits" in f] == [10] # the parameter's
    o = p.tx()(rp.Data(bits(80, 1), Fs=1e6, sps=1), fields={"signal": 20})
    assert [f["signal"] for _, f, _ in o.stream.control.fields("plcp") if "bits" in f] == [20] # the caller's
