"""What a profile needs before it runs (rp.profile.info, catalogue), and rp.profile.Decoder: a selected signal's samples
to a profile's messages"""
import json

import numpy as np
import pytest

import rpps as rp
from rpps.errors import ConfigError

def test_every_profile_says_what_it_needs():
    from rpps import _config
    cat = rp.profile.catalogue()
    assert {i.name for i in cat} == {f"{f}/{m}" for f, ms in rp.profile.ls().items() for m in ms}
    shipped = [i for i in cat if _config.PATH_CONFIG in _config.find("profile", i.name).parents] # (not a test's own)
    assert len(shipped) >= 19
    for i in shipped:
        r = i.requires
        assert r["input"] in ("baseband", "audio") and r["min_samples"] > 0
        if r["baud"] is None: # A rate the standard leaves open: Fs and bw wait for it
            assert r["Fs"] is None and r["bw"] is None
        else:
            assert r["Fs"] >= r["bw"] > 0
        assert i.kinds or i.name.startswith(("ieee80211b/1m", "ieee80211b/2m", "ieee802154/")) # (no frame step of their own)
        json.dumps(i.to_dict())

def test_requires_follow_the_params_and_baud():
    assert rp.profile.info("lte/downlink", n_rb=6).requires["Fs"] == 1.92e6
    assert rp.profile.info("adsb/mode_s", sps=2).requires["Fs"] == 4e6
    r = rp.profile.info("dvbs2/normal", baud=27.5e6).requires
    assert r["Fs"] == 4*27.5e6 and r["bw"] == pytest.approx(1.35*27.5e6)
    assert rp.profile.info("ble/adv").requires["from_cf"]["channel"]
    with pytest.raises(ValueError, match="rate is set"):
        rp.profile.info("ble/adv", baud=2e6)
    with pytest.raises(ConfigError, match="no params"):
        rp.profile.info("ble/adv", nope=1)

# ---- the Decoder

from rpps.signal.sample.fractional import fractional_resample

def adsb_capture(rng, frames=3):
    p = rp.profile.load("adsb/mode_s")
    payloads = [rng.integers(0, 2, 83).astype(bool) for _ in range(frames)]
    tx, s, waves, starts, at = p.tx(), None, [], [], 4000
    for i, m in enumerate(payloads):
        d = rp.Data(m, Fs=2e6, sps=1, stream=s, flush=i == frames - 1)
        s = d.stream
        o = tx(d, fields={"df": 17})
        starts.append(at)
        waves += [o.data*0.5, np.zeros(3000, np.complex64)]
        at += len(o.data) + 3000
    x = np.concatenate([np.zeros(4000, np.complex64)] + waves + [np.zeros(4000, np.complex64)])
    return x, payloads, np.array(starts)

def foreign(x, Fs, ratio, shift):
    """A capture at another rate (Fs*ratio), the signal `shift` Hz off centre, a little noise"""
    y = fractional_resample(x, ratio, Fs)
    y = y*np.exp(2j*np.pi*shift/(Fs*ratio)*np.arange(len(y)))
    rng = np.random.default_rng(9)
    return (y + 0.01*(rng.normal(size=len(y)) + 1j*rng.normal(size=len(y)))).astype(np.complex64)

def test_adsb_at_another_rate_and_offset():
    x, payloads, starts = adsb_capture(np.random.default_rng(1))
    y = foreign(x, 2e6, 1.2, 150e3) # 2.4 MS/s, the channel at +150 kHz
    dec = rp.profile.Decoder("adsb/mode_s", Fs=2.4e6, offset=150e3, cf=1090e6 - 150e3)
    msgs = []
    for a in range(0, len(y), 5000):
        msgs += dec.feed(y[a:a + 5000], start=a, time=100.0 + a/2.4e6)
    msgs += dec.close()
    good = [m for m in msgs if m.crc_ok]
    assert [m.payload for m in good] == [np.packbits(b).tobytes() for b in payloads]
    for m, s in zip(good, starts*1.2): # In the caller's index (2.4 MS/s), after the preamble
        assert 0 <= m.sample - s <= 1.2*(16 + 12)
        assert m.time == pytest.approx(100.0 + m.sample/2.4e6, abs=1e-6) and m.freq == pytest.approx(1090e6)
    st = dec.status
    assert st["state"] == "locked" and st["good"] == 3 and st["Fs"] == 2e6 and st["load"] is not None
    json.dumps(st)

def test_blocks_with_gaps_and_overlaps():
    """A display's reads: overlapping blocks are cut, a short gap filled, a long one starts over; frames clear of them all
    decode, placed in the caller's index"""
    x, payloads, starts = adsb_capture(np.random.default_rng(2), frames=6)
    dec = rp.profile.Decoder("adsb/mode_s", Fs=2e6, fill=0.001)
    msgs, a = [], 0
    for n, step in ((3000, 3000), (3000, 2500), (3000, 3000), (2000, 3500), (6000, 6000), (4000, 9000), (40000, 40000)):
        msgs += dec.feed(x[a:a + n], start=a)
        a += step # step < n: overlap; step > n: a gap (500 filled; 5000 > fill: a new stream)
    good = [m for m in msgs if m.crc_ok]
    want = {s: np.packbits(b).tobytes() for s, b in zip(starts, payloads)}
    assert good and all(any(0 <= m.sample - s <= 28 and m.payload == p for s, p in want.items()) for m in good)
    assert len({m.sample for m in good}) == len(good) # (no frame twice from an overlap)

def test_a_rate_too_low_is_refused():
    with pytest.raises(ValueError, match="needs"):
        rp.profile.Decoder("ble/adv", Fs=1e6)

def test_ble_channel_from_the_carrier():
    dec = rp.profile.Decoder("ble/adv", Fs=8e6, offset=1e6, cf=2425e6)
    assert dec.params["channel"] == 38 # 2426 MHz
    assert rp.profile.Decoder("ble/adv", Fs=8e6, offset=1e6, cf=2425e6, channel=37).params["channel"] == 37

def test_dvbs_rate_measured_then_decoded():
    rng = np.random.default_rng(5)
    p = rp.profile.load("dvbs/qpsk_1_2")
    pk = rng.integers(0, 256, (8*40, 188)).astype(np.uint8)
    pk[:, 0] = 0x47
    wave = p.tx()(rp.Data(pk.reshape(-1), Fs=250e3, sps=1, flush=True)) # 250 kBd at 4 samples a symbol: 1 MS/s
    y = foreign(np.asarray(wave.data), 1e6, 2.0, -300e3)                # at 2 MS/s, the carrier at -300 kHz
    dec = rp.profile.Decoder("dvbs/qpsk_1_2", Fs=2e6, offset=-300e3, bw=350e3)
    msgs, states = [], []
    for a in range(0, len(y), 50000):
        msgs += dec.feed(y[a:a + 50000], start=a)
        states.append(dec.status["state"])
    msgs += dec.close()
    assert states[0] == "measuring" and dec.status["baud"] == pytest.approx(250e3, rel=1e-3)
    good = [m for m in msgs if m.crc_ok and m.kind == "packet"]
    assert len(good) >= len(pk) - 40
    rows = {r.tobytes() for r in pk}
    assert all(m.payload in rows for m in good[1:])

def test_lte_through_a_resampler():
    from rpps.grid import lte
    sib = np.random.default_rng(2).integers(0, 2, 224).astype(np.uint8)
    x = lte.tx.channel(lte.tx.downlink(pci=10, ports=1, n_rb=6, frames=4, sib=sib), cfo=0.0, fs=1.92e6, delay=700,
                       snr_db=25, seed=3)
    y = foreign(x, 1.92e6, 2.0, 400e3) # 3.84 MS/s, the cell 400 kHz up
    dec = rp.profile.Decoder("lte/downlink", Fs=3.84e6, offset=400e3, n_rb=6)
    msgs = []
    for a in range(0, len(y), 40000):
        msgs += dec.feed(y[a:a + 40000], start=a)
    sib1 = [m for m in msgs if m.kind == "pdsch" and m.crc_ok]
    assert sib1 and all(m.payload == np.packbits(sib).tobytes() for m in sib1)
    assert any(m.kind == "pbch" and m.crc_ok and m.fields["pci"] == 10 for m in msgs)
    assert dec.status["state"] == "locked"
    mibs = [m for m in msgs if m.kind == "pbch"]
    dec = rp.profile.Decoder("lte/downlink", Fs=3.84e6, offset=400e3, n_rb=6, kinds={"pbch"}) # (the MIB alone: no PDSCH)
    only = []
    for a in range(0, len(y), 40000):
        only += dec.feed(y[a:a + 40000], start=a)
    assert [(m.sample, m.fields) for m in only] == [(m.sample, m.fields) for m in mibs]
    assert dec.status["state"] == "locked" and dec.status["messages"] == len(mibs)
    with pytest.raises(ConfigError, match="no message kinds"):
        rp.profile.Decoder("lte/downlink", Fs=3.84e6, n_rb=6, kinds={"sib1"})

def test_ax25_from_an_fm_channel():
    """AX.25 takes an FM channel's audio: the Decoder demodulates the channel before the profile"""
    rng = np.random.default_rng(13)
    p = rp.profile.load("ax25/afsk1200")
    pk = [bytes(rng.integers(0, 256, n).astype(np.uint8)) for n in (20, 30)]
    tx, s, audio = p.tx(), None, []
    for i, f in enumerate(pk):
        d = rp.Data(f, Fs=1200.0, sps=1, stream=s, flush=i == len(pk) - 1)
        s = d.stream
        o = tx(d) # The tones about 0 Hz (transmit leaves out `centre`)
        audio.append(np.asarray(o.data))
    x = np.concatenate([np.zeros(4800, np.complex64)] + audio + [np.zeros(4800, np.complex64)])
    a = np.real(x*np.exp(2j*np.pi*1700/48e3*np.arange(len(x)))) # The audio: the tones at 1200 and 2200 Hz
    a = fractional_resample(a/np.max(np.abs(a)), 240e3/48e3, 48e3).real # Audio at 240 kHz
    ph = np.cumsum(2*np.pi*3e3/240e3*a) # 3 kHz deviation
    y = (np.exp(1j*ph)*np.exp(2j*np.pi*50e3/240e3*np.arange(len(a)))).astype(np.complex64) # The channel 50 kHz up
    dec = rp.profile.Decoder("ax25/afsk1200", Fs=240e3, offset=50e3)
    msgs = []
    for k in range(0, len(y), 24000):
        msgs += dec.feed(y[k:k + 24000])
    msgs += dec.close()
    assert [m.payload for m in msgs if m.crc_ok] == pk
