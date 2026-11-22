"""Burst transmission: rp.sync.BurstTx places bursts on one stream (gaps, start times, ramps, a carrier each), a profile
with a burst step sends a burst a call (BurstTxRunner), and its receive side takes them back: one carrier, MF-TDMA
carriers sharing a slot, a hopping carrier, and a preamble the profile attaches itself"""
import numpy as np
import pytest

import rpps as rp
from rpps.errors import ConfigError

FS = 1e6

def burst(n, value=1.0, Fs=FS):
    return rp.Data(np.full(n, value, np.complex64), Fs=Fs, flush=True)

def whole(btx, blocks):
    return np.concatenate([np.asarray(b.data) for b in blocks])

# ---- BurstTx alone

def test_gaps_and_starts():
    btx = rp.sync.BurstTx(gap_s=100e-6)
    btx.add(burst(50))
    btx.add(burst(30))
    btx.add(burst(20), at=500e-6)
    a = btx.take()
    b = btx.take(flush=True)
    assert [s for s, _, _ in btx.sent] == [100, 250, 500]
    x = whole(btx, [a, b])
    assert len(a) == 500 and a.meta.pos == 0 and b.meta.pos == 500 and b.flush
    assert len(x) == 620 # 20 after 500, then the gap
    on = np.abs(x) > 0.5
    assert np.array_equal(np.nonzero(np.diff(on.astype(int)))[0] + 1, [100, 150, 250, 280, 500, 520])

def test_out_of_order_and_outside():
    btx = rp.sync.BurstTx()
    btx.add(burst(10), at=1e-3)
    with pytest.raises(ValueError, match="order"):
        btx.add(burst(10), at=0.5e-3)
    with pytest.raises(ValueError, match="outside"):
        btx.add(burst(10), freq=600e3)
    with pytest.raises(ValueError, match="no burst"):
        rp.sync.BurstTx().take()

def test_ramp():
    btx = rp.sync.BurstTx(ramp_s=10e-6)
    btx.add(burst(100))
    x = np.abs(whole(btx, [btx.take(flush=True)]))
    assert np.all(np.diff(x[:10]) > 0) and np.all(np.diff(x[-10:]) < 0) and x[0] < 0.05
    assert np.allclose(x[10:90], 1) and np.allclose(x[:10], x[::-1][:10])

def test_carriers_add_and_keep_their_phase():
    btx = rp.sync.BurstTx(Fs=4e6)
    btx.add(burst(256), at=0.0, freq=1e6)
    btx.add(burst(256), at=0.0, freq=-1e6) # Two carriers in one slot
    btx.add(burst(256), at=2e-3, freq=1e6)
    x = whole(btx, [btx.take(flush=True)])
    seg = x[100:900] # Inside the first slot: both tones, equal
    f = np.fft.fftfreq(len(seg), 1/4e6)
    S = np.abs(np.fft.fft(seg))
    top = sorted(f[np.argsort(S)[-2:]])
    assert np.allclose(top, [-1e6, 1e6], atol=4e6/len(seg))
    n = np.arange(len(x)) # Every burst on 1 MHz is one continuous carrier: the mixer's phase from sample 0
    a, b = x[200:600]*np.exp(-2j*np.pi*1e6/4e6*n[200:600]), x[8200:8600]*np.exp(-2j*np.pi*1e6/4e6*n[8200:8600])
    assert abs(np.angle(np.mean(a))) < 0.05 and abs(np.angle(np.mean(b))) < 0.05

def test_resampled_power_and_place():
    btx = rp.sync.BurstTx(Fs=3*FS)
    btx.add(burst(300), at=100e-6)
    x = whole(btx, [btx.take(flush=True)])
    on = np.nonzero(np.abs(x) > 0.5)[0]
    assert abs(on[0] - 300) <= 2 and abs(on[-1] - 1200) <= 3
    assert np.mean(np.abs(x[350:1150])**2) == pytest.approx(1.0, rel=0.02)

def test_streamed_as_a_whole():
    rng = np.random.default_rng(3)
    sizes, ats = [40, 90, 15, 60], [None, 300e-6, None, 900e-6]
    data = [rp.Data((rng.standard_normal(n) + 1j*rng.standard_normal(n)).astype(np.complex64), Fs=FS, flush=True) for n in sizes]
    one = rp.sync.BurstTx(gap_s=20e-6, ramp_s=3e-6)
    for d, at in zip(data, ats):
        one.add(d, at)
    ref = whole(one, [one.take(flush=True)])
    many, blocks = rp.sync.BurstTx(gap_s=20e-6, ramp_s=3e-6), []
    for d, at in zip(data, ats):
        many.add(d, at)
        blocks.append(many.take())
    blocks.append(many.take(flush=True))
    assert np.array_equal(whole(many, blocks), ref)
    assert [b.meta.pos for b in blocks] == list(np.cumsum([0] + [len(b) for b in blocks[:-1]]))

# ---- A profile's bursts

def ble_pdus(n=3, seed=5):
    rng = np.random.default_rng(seed)
    return [rng.integers(0, 2, 8*k).astype(bool) for k in (6, 20, 37)[:n]]

def pdus_of(outs):
    return [np.asarray(o.data).astype(bool) for o in outs if len(o)]

def test_ble_bursts_streamed_by_call():
    p, pdus = rp.profile.load("ble/adv"), ble_pdus()
    tx = p.tx()
    blocks = [tx(rp.Data(pdu, Fs=1e6, sps=1), fields={"type": 2}) for pdu in pdus] + [tx.flush()]
    x = whole(None, blocks)
    assert [s for s, _, _ in tx.sent][0] == 4000 and tx.bursts == 3
    outs = p.rx()(rp.Data(x, Fs=4e6, sps=4, flush=True))
    got = pdus_of(outs)
    assert len(got) == 3 and all(np.array_equal(a, b) for a, b in zip(got, pdus))

def receive_carrier(p, x, Fs, f):
    """The PDUs of the bursts on carrier f of a wideband stream: brought to 4 MS/s at baseband, then the profile"""
    chain, Fs_out, _ = rp.signal.channelize(f, 2e6, Fs, oversample=2.0)
    assert Fs_out == 4e6
    y = rp.Data(x.astype(np.complex64), Fs=Fs, flush=True) + chain
    return pdus_of(p.rx()(rp.Data(np.asarray(y.data, np.complex64), Fs=4e6, sps=4, flush=True)))

def test_mf_tdma_carriers_in_one_slot():
    p, pdus = rp.profile.load("ble/adv"), ble_pdus(2)
    tx = p.tx()
    tx.burst_tx.Fs = 12e6
    tx.burst_tx.reset()
    o = tx([rp.Data(pdu, Fs=1e6, sps=1) for pdu in pdus], at=[1e-3, 1e-3], freq=[-4e6, 4e6], flush=True)
    assert o.meta.Fs == 12e6 and [s for s, _, _ in tx.sent] == [12000, 12000]
    x = np.asarray(o.data)
    for pdu, f in zip(pdus, (-4e6, 4e6)):
        got = receive_carrier(p, x, 12e6, f)
        assert len(got) == 1 and np.array_equal(got[0], pdu)

def test_hopping_carrier():
    obj = rp._config.read(rp._config.find("profile", "ble/adv"))
    obj["steps"][0].update({"Fs": 12e6, "hop": [-4e6, 0.0, 4e6]})
    p, pdus = rp._config.load_obj("profile", "ble/hop", obj), ble_pdus(3)
    tx = p.tx()
    x = np.asarray(tx([rp.Data(pdu, Fs=1e6, sps=1) for pdu in pdus], flush=True).data)
    assert [f for _, _, f in tx.sent] == [-4e6, 0.0, 4e6]
    for pdu, f in zip(pdus, (-4e6, 0.0, 4e6)):
        got = receive_carrier(p, x, 12e6, f)
        assert len(got) == 1 and np.array_equal(got[0], pdu)

def test_bad_hop():
    obj = rp._config.read(rp._config.find("profile", "ble/adv"))
    obj["steps"][0]["hop"] = []
    with pytest.raises(ConfigError, match="hop"):
        rp._config.load_obj("profile", "ble/hop", obj).tx()

# ---- A preamble the profile attaches

def attach_profile(**pre):
    step = {"sync": "preamble", "id": "pre", "name": "pre", "word": "0x1acffc1d", "bits": 32, "gate": 0.8, "attach": True}
    step = {k: v for k, v in {**step, **pre}.items() if v is not None}
    return rp._config.load_obj("profile", "test/attach", {
        "base": "profile",
        "steps": [
            {"sync": "burst", "win": 8, "gap_s": 200e-6},
            step,
            {"wave": "shape", "pulse": "rrc/0p35", "sps": 4},
            {"sync": "timing", "method": "known", "anchor": "pre"},
            {"symbol": "constellation", "name": "psk/bpsk"},
        ],
    })

PRE = np.unpackbits(np.frombuffer(bytes.fromhex("1acffc1d"), np.uint8)).astype(bool)

def test_preamble_attached_and_found():
    p = attach_profile()
    rng = np.random.default_rng(8)
    payloads = [rng.integers(0, 2, n).astype(bool) for n in (100, 240)]
    tx = p.tx()
    assert "PreambleTx" in [type(op).__name__ for _, op in tx]
    x = np.asarray(tx([rp.Data(b, Fs=1e6, sps=1) for b in payloads], flush=True).data)
    assert tx.sent[0][1] > (32 + 100)*4 # The preamble's symbols sent before the payload's
    outs = p.rx()(rp.Data(x.astype(np.complex64), Fs=4e6, sps=4, flush=True))
    assert len(outs) == 2
    for o, b in zip(outs, payloads):
        got = np.asarray(o.data)
        got = got < 0 if got.dtype.kind == "f" else got.astype(bool)
        assert np.array_equal(got[:32 + len(b)], np.r_[PRE, b]) # (the output starts at the preamble found)
        assert o.stream.est["pre"]["metric"] > 0.8

def test_attach_needs_a_waveform():
    with pytest.raises(ConfigError, match="attach"):
        attach_profile(word=None, bits=None, repeat=16, window=64)

def test_attach_tx_sends_its_preambles_template():
    """The transmit side reads the Preamble's template when a stream starts: set after it was made, set again, it follows"""
    pre = rp.sync.Preamble(pending=True)
    ptx = pre.attach_tx()
    t = np.array([1.0, 2.0, 3.0, 0.5, 0.25])
    pre.set_template(t, span=3) # 3 samples of symbols, then a tail of 2
    out = np.asarray((rp.Data(np.ones(4, np.float32), flush=True) + ptx).data)
    assert np.allclose(out, [1, 2, 3, 1.5, 1.25, 1, 1])
    pre.set_template(-t) # span None: all of it before the stream
    out = np.asarray((rp.Data(np.ones(2, np.float32), flush=True) + ptx).data)
    assert np.allclose(out, [-1, -2, -3, -0.5, -0.25, 1, 1])
