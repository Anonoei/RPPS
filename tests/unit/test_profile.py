"""Profiles: a whole signal from one config file (ARCHITECTURE.md, Profiles; Migration step 6)"""
import numpy as np
import pytest

import rpps as rp
from rpps.base.types import Rep
from rpps.errors import ConfigError

KNOWN = {"timing": {"method": "known"}} # loopback: the timing is known, nothing to estimate

def hard(d):
    x = np.asarray(d.data)
    return x < 0 if x.dtype.kind == "f" else x.astype(bool)

def names(runner):
    return [f"{how}{type(op).__name__}" for how, op in runner]

def test_listed():
    assert {"ieee802154", "ieee80211b"} <= set(rp.profile.ls())

def test_802154_builds_the_documented_pairs():
    p = rp.profile.load("ieee802154/2450", sync=KNOWN)
    assert [f"{s.category}:{s.kind}" for s in p.steps] == ["sync:carrier", "wave:stagger", "wave:shape", "sync:timing", "symbol:spread"]
    assert names(p.rx()) == ["+Stagger", "+Designed", "+_KnownTiming", "/Spread"] # carrier: method none
    assert names(p.tx()) == ["*Spread", "+Zeros", "+Designed", "+Stagger"]
    assert str(p.rx().ops[0][1]) == "Stagger(i)" and str(p.tx().ops[-1][1]) == "Stagger(q)"
    assert [s.takes for s in p.steps] == [Rep.SAMPLES, Rep.SAMPLES, Rep.SAMPLES, Rep.SAMPLES, Rep.SYMBOLS]

@pytest.mark.parametrize("sps", [2, 4, 8])
def test_802154_loopback(sps):
    rng = np.random.default_rng(sps)
    p = rp.profile.load("ieee802154/2450", sync=KNOWN, sps=sps)
    msg = rng.integers(0, 256, 60).astype(np.uint8)
    wave = p.tx()(rp.Data(msg, Fs=250e3, sps=1))
    assert wave.REP == Rep.SAMPLES and wave.sps == sps
    out = p.rx()(wave)
    assert out.REP == Rep.CHANNEL
    got, bits = hard(out), np.unpackbits(msg).astype(bool)
    assert len(got) >= len(bits) - 8 and np.array_equal(got, bits[:len(got)])

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

def test_any_block_cut_on_receive():
    rng = np.random.default_rng(2)
    p = rp.profile.load("ieee80211b/2m", sync=KNOWN)
    wave = p.tx()(rp.Data(rng.integers(0, 256, 30).astype(np.uint8), Fs=1e6, sps=1))
    ref = hard(p.rx()(rp.Data(wave.data, Fs=wave.Fs, sps=wave.sps, delay=wave.delay)))
    for cuts in ([1000, 3, len(wave) - 1003], [97]*20 + [len(wave) - 1940]):
        rx, s, outs, i = p.rx(), None, [], 0
        for b in cuts:
            d = rp.Data(wave.data[i:i + b], Fs=wave.Fs, sps=wave.sps, delay=wave.delay, stream=s)
            s = d.stream
            outs.append(hard(rx(d)))
            i += b
        got = np.concatenate(outs)
        assert np.array_equal(got, ref[:len(got)]) and len(got) >= len(ref) - 64

def test_any_block_cut_on_transmit():
    rng = np.random.default_rng(3)
    p = rp.profile.load("ieee802154/2450", sync=KNOWN)
    msg = rng.integers(0, 256, 40).astype(np.uint8)
    whole = p.tx()(rp.Data(msg, Fs=250e3, sps=1)).data
    tx, s, outs = p.tx(), None, []
    for a, b in ((0, 7), (7, 8), (8, 40)):
        d = rp.Data(msg[a:b], Fs=250e3, sps=1, stream=s)
        s = d.stream
        outs.append(tx(d).data)
    got = np.concatenate(outs)
    assert np.allclose(got, whole[:len(got)], atol=1e-6) and len(got) >= len(whole) - 64

def test_a_source_of_chips_skips_the_wave_and_timing_steps():
    """A source that hands over timed chips (one per sample) starts the receiver at SYMBOLS"""
    rng = np.random.default_rng(4)
    p = rp.profile.load("ieee80211b/1m")
    msg = rng.integers(0, 256, 20).astype(np.uint8)
    tx_ops = [op for how, op in p.tx() if how == "*"] # the symbol and code steps only: chips at one per chip
    d = rp.Data(msg)
    for op in tx_ops:
        d = d * op
    rx = p.rx(start=Rep.SYMBOLS)
    assert [how for how, _ in rx] == ["/"]*4 and type(rx.ops[0][1]).__name__ == "Spread" # no wave, no timing
    out = rx(rp.Data(d.data, sps=1))
    got, bits = hard(out), np.unpackbits(msg).astype(bool)
    assert np.array_equal(got[8:], bits[8:len(got)])

def test_applied_sync_is_skipped():
    p = rp.profile.load("ieee80211b/1m")
    assert "+OerderMeyr" in names(p.rx()) and "+OerderMeyr" not in names(p.rx(applied={"timing"}))

def test_defaults_and_overrides():
    assert "+OerderMeyr" in names(rp.profile.load("ieee80211b/1m").rx()) # registry default: first class (step 8)
    assert "+MPower" in names(rp.profile.load("ieee80211b/long").rx())
    p = rp.profile.load("ieee80211b/1m", sync={"timing": {"method": "gardner"}}) # the loops are there by name
    assert p.rx().ops[1][1].ted == "gardner"

def test_errors():
    with pytest.raises(ConfigError):
        rp.profile.load("ieee80211b/1m", sync={"timing": {"method": "telepathy"}})
    with pytest.raises(ConfigError):
        rp.profile.Profile("bad", {"steps": [{"symbol": "constellation", "code": "fec"}]})
    with pytest.raises(ConfigError):
        rp.profile.Profile("bad", {"steps": [{"wave": "warp"}]})
    with pytest.raises(ConfigError):
        rp.profile.Profile("bad", {"steps": []})
    with pytest.raises(FileNotFoundError):
        rp.profile.load("nothing/here")

# ---- DVB-S: what `chain` did, as a profile of ordinary steps

DVBS_KNOWN = {"timing": {"method": "known"}, "carrier": {"method": "none"}}

def ts(rng, n):
    pk = rng.integers(0, 256, (n, 188)).astype(np.uint8)
    pk[:, 0] = 0x47
    return pk

def dvbs_packets(out):
    b = np.packbits(hard(out))
    return b[:len(b)//188*188].reshape(-1, 188)

def first_match(got, pk, startup=1):
    """(output index, input index) of the first output packet found in the input; the first `startup` outputs may carry
    the decoder's start-up errors (a Viterbi that starts mid-trellis)"""
    for i in range(min(startup + 1, len(got))):
        k = next((k for k in range(len(pk)) if np.array_equal(got[i], pk[k])), None)
        if k is not None:
            return i, k
    return None, None

@pytest.mark.parametrize("turn", [1, -1]) # -1: the carrier 180 degrees round: every decoded bit inverted, and found
def test_dvbs_round_trip(turn):
    rng = np.random.default_rng(5)
    p = rp.profile.load("dvbs/qpsk_3_4", sync=DVBS_KNOWN)
    pk = ts(rng, 8*12)
    wave = p.tx()(rp.Data(pk.reshape(-1), Fs=1e6, sps=1, flush=True))
    out = p.rx()(rp.Data(turn*wave.data, Fs=wave.Fs, sps=wave.sps, delay=wave.delay, flush=True))
    got = dvbs_packets(out)
    est = out.stream.est
    assert est["packets"]["polarity"] == turn and est["frames"]["locked"]
    i, k = first_match(got, pk, startup=0)
    assert i == 0 and len(got) >= len(pk) - 16 # latency: the deinterleaver's 11 packets, the Viterbi tail
    assert np.array_equal(got, pk[k:k + len(got)])

def test_dvbs_finds_the_puncture_phase():
    """A receiver that starts mid-stream on channel bits: two coded bits missing, so the puncture pattern starts elsewhere"""
    rng = np.random.default_rng(6)
    p = rp.profile.load("dvbs/qpsk_3_4", sync=DVBS_KNOWN)
    pk = ts(rng, 8*10)
    tx_ops = [op for how, op in p.tx() if how == "*"][:-1] # the code steps: data bits to channel bits (no QPSK)
    d = rp.Data(pk.reshape(-1), flush=True)
    for op in tx_ops:
        d = d * op
    coded = np.asarray(d.data).astype(bool)[2:]
    rx = p.rx(start=Rep.CHANNEL)
    out = rx(rp.Data(coded, flush=True).derive(coded, rp.Category.CODE, rp.dtype.BITS, rep=Rep.CHANNEL))
    assert out.stream.est["puncture"]["offset"] != 0
    got = dvbs_packets(out)
    i, k = first_match(got, pk)
    assert i is not None and len(got) - i >= len(pk) - 16
    assert np.array_equal(got[i:], pk[k:k + len(got) - i])

def test_profile_parameters_reach_the_steps(tmp_path):
    """What `chain` did with its params: a profile's parameter goes into each step that uses it"""
    from rpps._config import format_json
    d = tmp_path / "profile" / "test"
    d.mkdir(parents=True)
    (d / "ble_whitening.json").write_text(format_json({"base": "profile", "params": {"channel": {"default": 37}},
        "steps": [{"code": "scramble", "name": "ble_whiten", "args": {"channel": {"param": "channel"}}}]}))
    rp.add_config_path(tmp_path)
    try:
        p = rp.profile.load("test/ble_whitening", channel=38)
        s = p.steps[0].obj
        assert np.array_equal(s.sequence(50), rp.code.scram.load("ble_whiten", channel=38).sequence(50))
    finally:
        rp._config._EXTRA.remove(tmp_path)
