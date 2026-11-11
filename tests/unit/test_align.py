"""Sync on bits: Align (ARCHITECTURE.md, Migration step 3)"""
import numpy as np
import pytest

import rpps as rp
from rpps.base.types import Rep
from rpps.errors import StreamError

P = 204*8 # bits per DVB-S packet
DVBS = [0xB8] + [0x47]*7 # what starts each packet in turn

def ts_bits(rng, packets):
    """DVB-S-like: 204-byte packets, the first byte 0x47, or 0xB8 every eighth"""
    pk = rng.integers(0, 256, (packets, 204)).astype(np.uint8)
    pk[:, 0] = 0x47
    pk[::8, 0] = 0xB8
    return np.unpackbits(pk.reshape(-1)).astype(bool)

def run(x, cuts, stage, rep=Rep.CHANNEL):
    s, outs, i = None, [], 0
    for b in cuts:
        d = rp.Data(x[i:i + b]) if x.dtype == bool else rp.base.Mod(x[i:i + b])
        d = d.derive(d.data, d.CAT, d.DT, ctx={}, meta=d.meta if s is None else rp.base.Meta(stream=s), rep=rep)
        if s is None:
            s = d.stream
        out = d + stage
        assert out.REP == rep and out.DT == d.DT
        outs.append(out.data)
        i += b
    return np.concatenate(outs), s

def cuts_of(n, rng):
    c, left = [], n
    while left:
        b = int(min(left, rng.integers(1, 5000)))
        c.append(b)
        left -= b
    return c

@pytest.mark.parametrize("prefix", [0, 37, 1000])
def test_aligns_and_drops_what_comes_before(prefix):
    rng = np.random.default_rng(prefix)
    bits = ts_bits(rng, 40)
    x = np.concatenate((rng.integers(0, 2, prefix).astype(bool), bits))
    out, s = run(x, [len(x)], rp.sync.Align(204, pattern=DVBS))
    assert len(out) % P == 0 and len(out) >= len(bits) - 2*P
    k = (len(bits) - len(out))//P # whole packets lost to the prefix not being a whole packet
    assert np.array_equal(out, bits[k*P:k*P + len(out)])
    est = s.est["align"]
    assert est["locked"] and est["polarity"] == 1 and est["fraction"] >= 0.9 and est["dropped"] == prefix + k*P

def test_inverted_polarity_is_put_right():
    rng = np.random.default_rng(1)
    bits = ts_bits(rng, 30)
    x = ~np.concatenate((rng.integers(0, 2, 11).astype(bool), bits))
    out, s = run(x, [len(x)], rp.sync.Align(204, pattern=DVBS))
    assert s.est["align"]["polarity"] == -1
    assert np.array_equal(out, bits[:len(out)])

def test_llrs_stay_llrs_and_are_negated_for_polarity():
    rng = np.random.default_rng(2)
    bits = ts_bits(rng, 30)
    llr = (1 - 2*bits.astype(np.float32))*3 + rng.normal(0, 1.0, len(bits)).astype(np.float32)
    x = -np.concatenate((rng.normal(0, 3, 50).astype(np.float32), llr))
    out, s = run(x, [len(x)], rp.sync.Align(204, pattern=DVBS))
    assert out.dtype.kind == "f" and s.est["align"]["polarity"] == -1
    assert np.mean((out < 0) != bits[:len(out)]) < 0.01

def test_any_block_cut():
    rng = np.random.default_rng(3)
    bits = ts_bits(rng, 30)
    x = np.concatenate((rng.integers(0, 2, 500).astype(bool), bits))
    ref, _ = run(x, [len(x)], rp.sync.Align(204, pattern=DVBS))
    for seed in range(3):
        got, _ = run(x, cuts_of(len(x), np.random.default_rng(seed)), rp.sync.Align(204, pattern=DVBS))
        assert np.array_equal(got, ref)

def test_a_bit_slip_loses_lock_and_reacquires():
    rng = np.random.default_rng(4)
    bits = ts_bits(rng, 60)
    slip = 25*P + 100
    x = np.concatenate((bits[:slip], bits[slip + 3:])) # three bits lost
    out, s = run(x, cuts_of(len(x), rng), rp.sync.Align(204, pattern=DVBS))
    est = s.est["align"]
    assert est["locks"] == 2 and est["locked"]
    tail = out[-10*P:] # aligned again after the slip, on the original packets
    starts = [i for i in range(0, len(bits) - 10*P + 1, P) if np.array_equal(bits[i:i + 10*P], tail)]
    assert starts

def test_a_set_of_markers_cannot_tell_dvbs_polarity_but_the_pattern_can():
    rng = np.random.default_rng(5)
    x = ~ts_bits(rng, 30)
    _, s = run(x, [len(x)], rp.sync.Align(204, markers=(0x47, 0xB8)))
    assert s.est["align"]["polarity"] == 1 # inverted, 0x47 and 0xB8 swap: still all markers
    _, s = run(x, [len(x)], rp.sync.Align(204, pattern=DVBS))
    assert s.est["align"]["polarity"] == -1

def test_markers_set_mode():
    rng = np.random.default_rng(6)
    pk = rng.integers(0, 256, (40, 64)).astype(np.uint8)
    pk[:, 0] = 0x7E
    bits = np.unpackbits(pk.reshape(-1)).astype(bool)
    x = np.concatenate((rng.integers(0, 2, 13).astype(bool), bits))
    out, s = run(x, cuts_of(len(x), rng), rp.sync.Align(64, markers=(0x7E,), polarity=False))
    assert s.est["align"]["locked"] and len(out) % (64*8) == 0
    k = (len(bits) - len(out))//(64*8) # whole packets dropped before lock
    assert np.array_equal(out, bits[k*64*8:k*64*8 + len(out)])

def test_refuses_samples_and_stays_out_of_signals():
    with pytest.raises(StreamError):
        rp.Data(np.zeros(8, np.complex64)) + rp.sync.Align(4, markers=(0x47,))
    with pytest.raises(StreamError):
        rp.Data(np.zeros(8, bool)) + rp.signal.sample.down.Decimate(2) # signal stages still refuse bits
