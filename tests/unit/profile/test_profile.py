"""Profiles: a whole signal from one config file (rpps.profile)"""
import numpy as np
import pytest

import rpps as rp
from rpps.base.types import Rep
from rpps.errors import ConfigError
from support.signals import hard
from support.profiles import names
from support.dvbs2 import KNOWN

def test_listed():
    assert {"ieee802154", "ieee80211b"} <= set(rp.profile.ls())

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

def test_a_runner_adds_to_blocks_like_a_stage():
    """`src(n) + p.rx()` (the receive loop the guides use) gives what calling the runner on each block does"""
    rng = np.random.default_rng(2)
    p = rp.profile.load("ieee80211b/2m", sync=KNOWN)
    wave = p.tx()(rp.Data(rng.integers(0, 256, 30).astype(np.uint8), Fs=1e6, sps=1))
    src = lambda: rp.source.Memory(wave.data, Fs=wave.Fs, sps=wave.sps)(500)
    rx = p.rx()
    ref = np.concatenate([hard(rx(b)) for b in src()])
    got = np.concatenate([hard(out) for out in src() + p.rx()])
    assert len(ref) > 100 and np.array_equal(got, ref)
    rx = p.rx()
    assert np.array_equal(hard(rp.Data(wave.data, Fs=wave.Fs, sps=wave.sps) + rx),
                          hard(p.rx()(rp.Data(wave.data, Fs=wave.Fs, sps=wave.sps))))

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
    p = rp.profile.load("ieee80211b/1m", sync={"timing": {"method": "mat"}}) # another estimator by name
    assert "+MatTrack" in names(p.rx())

def test_the_timing_loops_are_there_by_name():
    p = rp.profile.load("ieee80211b/1m", sync={"timing": {"method": "gardner"}}) # second class (rpps.sync.loops)
    mm = next(op for _, op in p.rx().ops if isinstance(op, rp.sync.loops.MM))
    assert mm.ted == "gardner"

def test_mm_timing_step_takes_biphase():
    """{"sync": "timing", "method": "mm", "biphase": true}: M&M on Manchester symbols (RDS)"""
    p = rp.profile.Profile("x", {"steps": [{"sync": "timing", "method": "mm", "alpha": 0.05, "biphase": True},
                                           {"symbol": "constellation", "name": "psk/bpsk"}]})
    mm = p.rx().ops[0][1]
    assert isinstance(mm, rp.sync.loops.MM) and mm.biphase and mm.ted == "mm"

def test_errors():
    with pytest.raises(ConfigError):
        rp.profile.load("ieee80211b/1m", sync={"timing": {"method": "telepathy"}})
    with pytest.raises(ConfigError):
        rp.profile.Profile("bad", {"steps": [{"symbol": "constellation", "code": "conv"}]})
    with pytest.raises(ConfigError):
        rp.profile.Profile("bad", {"steps": [{"wave": "warp"}]})
    with pytest.raises(ConfigError):
        rp.profile.Profile("bad", {"steps": []})
    with pytest.raises(FileNotFoundError):
        rp.profile.load("nothing/here")

def test_profile_parameters_reach_the_steps(tmp_path, config_layer):
    """What `chain` did with its params: a profile's parameter goes into each step that uses it"""
    from rpps._config import format_json
    d = tmp_path / "profile" / "test"
    d.mkdir(parents=True)
    (d / "ble_whitening.json").write_text(format_json({"params": {"channel": {"default": 37}},
        "steps": [{"code": "scramble", "name": "additive/ble_whiten", "args": {"channel": {"param": "channel"}}}]}))
    config_layer(tmp_path)
    p = rp.profile.load("test/ble_whitening", channel=38)
    s = p.steps[0].obj
    z = np.zeros(50, bool)
    assert np.array_equal(s.encode(z), rp.code.scram.load("ble_whiten", channel=38).encode(z))

def test_domain_loader_keeps_the_given_parameters():
    """Loaded through the generic config loader, a profile gets the values given, not its defaults"""
    from rpps import _config
    assert _config.load("profile", "ieee802154/2450", params={"sps": 8}).params == {"sps": 8}
    assert rp.profile.load("ieee802154/2450", sps=8).params == {"sps": 8}
    assert rp.profile.load("ieee802154/2450").params == {"sps": 4}
