"""The packet and framed-telemetry profiles, from one config file each, transmit and receive: Bluetooth LE advertising (GFSK,
whitening, CRC-24), Mode S / ADS-B (pulse position, preamble, parity), AX.25 AFSK (NRZI, HDLC) and CCSDS telemetry (ASM,
randomizer, interleaved Reed-Solomon)"""
import numpy as np
import pytest

import rpps as rp
from rpps.base.types import Rep

KNOWN = {"timing": {"method": "known"}, "carrier": {"method": "none"}}

def hard(d):
    x = np.asarray(d.data)
    return x < 0 if x.dtype.kind == "f" else x.astype(bool)

def names(runner):
    return [f"{how}{type(op).__name__}" for how, op in runner]

def receive(p, x, Fs, sps, delay, cuts=None):
    rx, st, outs, i = p.rx(), None, [], 0
    cuts = cuts or [len(x)]
    for k, n in enumerate(cuts):
        d = rp.Data(x[i:i + n], Fs=Fs, sps=sps, delay=delay, stream=st, flush=k == len(cuts) - 1)
        st, i = d.stream, i + n
        outs.append(hard(rx(d)))
    return np.concatenate(outs), st

def receive_bursts(p, x, Fs, sps, cuts=None):
    """A burst profile: (the bits of every burst joined, the bursts' outputs)"""
    rx, st, outs, i = p.rx(), None, [], 0
    cuts = cuts or [len(x)]
    for k, n in enumerate(cuts):
        d = rp.Data(x[i:i + n], Fs=Fs, sps=sps, stream=st, flush=k == len(cuts) - 1)
        st, i = d.stream, i + n
        outs += rx(d)
    return (np.concatenate([hard(o) for o in outs]) if outs else np.zeros(0, bool)), outs

def total(outs, name, key):
    """An estimate counted over the bursts (each burst is a stream of its own)"""
    return sum(o.stream.est.get(name, {}).get(key, 0) for o in outs)

def test_listed():
    assert {"ble", "adsb", "ax25", "ccsds"} <= set(rp.profile.ls())

# ---- Bluetooth LE advertising

@pytest.fixture(scope="module")
def ble_signal():
    rng = np.random.default_rng(11)
    p = rp.profile.load("ble/adv")
    pdus = [rng.integers(0, 2, 8*n).astype(bool) for n in (6, 20, 37)]
    tx = p.tx() # A burst a call, 1 ms apart (the profile's gap_s: advertising events are milliseconds apart)
    o = tx([rp.Data(pdu, Fs=1e6, sps=1) for pdu in pdus], fields={"type": 2, "txadd": 1}, flush=True)
    return p, np.asarray(o.data), o, pdus

def test_ble_builds_the_pairs(ble_signal):
    p = ble_signal[0]
    assert names(p.rx()) == ["+Burst", "+FMDemod", "+Matched", "+Preamble", "+Known", "/Constellation", "+Delimiter", "/Marker",
                             "/additive/ble_whiten", "/Header"]
    assert names(p.tx()) == ["*Header", "*additive/ble_whiten", "*Marker", "*Constellation", "+Shape", "+FM", "+BurstTx"]

def test_ble_packets_round_trip(ble_signal):
    p, x, o, pdus = ble_signal
    got, outs = receive_bursts(p, x, o.Fs, 4)
    assert len(outs) == 3 and np.array_equal(got, np.concatenate(pdus)) # One output per packet
    assert total(outs, "ble_adv", "frames") == 3 and total(outs, "ble_adv", "fcs_failed") == 0 and total(outs, "aa", "frames") == 3
    assert outs[-1].stream.est["ble_adv"]["last"] == {"type": 2, "type.name": "ADV_NONCONN_IND", "rfu": 0, "chsel": 0,
                                                      "txadd": 1, "rxadd": 0, "length": 37}
    assert [o.meta.pos > 0 for o in outs] == [True]*3

def test_ble_any_block_cut(ble_signal):
    p, x, o, pdus = ble_signal
    got, _ = receive_bursts(p, x, o.Fs, 4, cuts=[333, 1, 700, 2600, len(x) - 3634])
    assert np.array_equal(got, np.concatenate(pdus))

def test_ble_packets_with_a_carrier_offset_and_noise(ble_signal):
    """Each packet on its own: the preamble step finds the access address in the discriminator's output, places the bit
    timing on it and takes the 120 kHz offset off (a level there of about half a symbol's)"""
    p, x, o, pdus = ble_signal
    rng = np.random.default_rng(5)
    n = len(x)
    noise = 0.05*(rng.standard_normal(n) + 1j*rng.standard_normal(n))
    levels = {}
    for cfo in (60e3, 120e3):
        y = (x*np.exp(2j*np.pi*cfo/o.Fs*np.arange(n)) + noise).astype(np.complex64)
        got, outs = receive_bursts(p, y, o.Fs, 4)
        assert np.array_equal(got, np.concatenate(pdus))
        levels[cfo] = np.array([ob.stream.est["aa_wave"]["level"] for ob in outs])
    assert (levels[120e3] > levels[60e3]).all() and (levels[60e3] > 0.1).all() # The offset, measured (its scale: the filters')

def test_ble_whitening_is_the_channel_s(ble_signal):
    """The same packets on another channel are whitened differently, and a receiver on the wrong channel sees no valid frame"""
    p, x, o, pdus = ble_signal
    wrong = rp.profile.load("ble/adv", channel=38)
    got, outs = receive_bursts(wrong, x, o.Fs, 4)
    assert len(got) == 0 and total(outs, "ble_adv", "frames") == 0 and total(outs, "aa", "frames") == 3

def test_ble_sync_word_is_not_whitened(ble_signal):
    p, x, o, pdus = ble_signal
    tx_ops = [op for how, op in p.tx() if how == "*"]
    d = rp.Data(pdus[0], Fs=1e6, sps=1)
    for op in tx_ops[:-1]: # header, whitening, marker: the bits as sent
        d = d * op
    w = rp.sequence.load("ble_adv_aa")
    assert np.array_equal(hard(d)[:40], w.bits)

# ---- Mode S / ADS-B

@pytest.fixture(scope="module")
def adsb_signal():
    rng = np.random.default_rng(12)
    p = rp.profile.load("adsb/mode_s")
    msgs = [rng.integers(0, 2, 83).astype(bool) for _ in range(3)]
    tx, s, waves = p.tx(), None, []
    for i, m in enumerate(msgs):
        d = rp.Data(m, Fs=2e6, sps=1, stream=s, flush=i == len(msgs) - 1)
        s = d.stream
        o = tx(d, fields={"df": 17})
        waves += [o.data*(0.3 + i*0.5), (0.02*(rng.normal(size=301) + 1j*rng.normal(size=301))).astype(np.complex64)]
    x = np.concatenate([np.zeros(25, np.complex64)] + waves)
    return p, x, o, msgs

def test_adsb_transmits_the_preamble_and_pulse_positions(adsb_signal):
    p, x, o, msgs = adsb_signal
    chips = np.abs(np.asarray(o.data)) > 0.5
    assert len(chips) == 16 + 2*112 and np.array_equal(chips[:16], rp.sequence.load("mode_s_preamble").bits)
    assert np.all(chips[16::2] != chips[17::2]) # one pulse per bit, in one half or the other
    assert np.array_equal(chips[16:16 + 10:2], [1, 0, 0, 0, 1]) # DF 17 = 10001

def test_adsb_frames_of_differing_level_round_trip(adsb_signal):
    p, x, o, msgs = adsb_signal
    got, st = receive(p, x, 2e6, 1, o.delay)
    assert np.array_equal(got, np.concatenate(msgs))
    assert st.est["mode_s"]["frames"] == 3 and st.est["mode_s"]["fcs_failed"] == 0 and st.est["mode_s"]["last"] == {"df": 17}

def test_adsb_any_block_cut(adsb_signal):
    p, x, o, msgs = adsb_signal
    got, _ = receive(p, x, 2e6, 1, o.delay, cuts=[100, 7, 500, len(x) - 607])
    assert np.array_equal(got, np.concatenate(msgs))

def test_adsb_damaged_frame_is_dropped(adsb_signal):
    p, x, o, msgs = adsb_signal
    y = x.copy()
    k = 25 + 16 + 40 # the two halves of one bit of the first frame's data swapped: that bit inverted
    y[k], y[k + 1] = y[k + 1], y[k]
    got, st = receive(p, y, 2e6, 1, o.delay)
    assert st.est["mode_s"]["fcs_failed"] == 1 and np.array_equal(got, np.concatenate(msgs[1:]))

def test_adsb_real_frame_decodes():
    """A recorded DF17 frame, modulated by the profile's own transmitter at a different level, decodes to its fields"""
    frame = bytes.fromhex("8D4840D6202CC371C32CE0576098")
    bits = np.unpackbits(np.frombuffer(frame, np.uint8)).astype(bool)
    p = rp.profile.load("adsb/mode_s")
    o = p.tx()(rp.Data(bits[5:88], Fs=2e6, sps=1, flush=True), fields={"df": 17})
    got, st = receive(p, np.concatenate((np.zeros(9, np.complex64), 0.4*o.data, np.zeros(9, np.complex64))), 2e6, 1, o.delay)
    assert np.array_equal(got, bits[5:88]) and st.est["mode_s"]["last"] == {"df": 17}

# ---- AX.25 AFSK 1200

@pytest.fixture(scope="module")
def ax25_signal():
    rng = np.random.default_rng(13)
    p = rp.profile.load("ax25/afsk1200", centre=0, sps=10)
    pk = [bytes(rng.integers(0, 256, n).astype(np.uint8)) for n in (20, 45, 30)]
    tx, s, waves = p.tx(), None, []
    for i, f in enumerate(pk):
        d = rp.Data(f, Fs=1200.0, sps=1, stream=s, flush=i == len(pk) - 1)
        s = d.stream
        o = tx(d)
        waves.append(o.data)
    x = np.concatenate([np.zeros(400, np.complex64)] + waves + [np.zeros(400, np.complex64)])
    x = (x + 0.05*(rng.normal(size=len(x)) + 1j*rng.normal(size=len(x)))).astype(np.complex64)
    return p, x, o, pk

def test_ax25_builds_the_pairs(ax25_signal):
    p = ax25_signal[0]
    assert names(p.rx()) == ["+ShiftFreq", "+Designed", "+FMDemod", "+Matched", "+OerderMeyr", "/Constellation", "/Line", "/ax25"]
    assert names(p.tx()) == ["*ax25", "*Line", "*Constellation", "+Shape", "+FM"]
    assert p.rx().ops[4][1].min_symbols == 256 and p.rx().ops[4][1].gate == 0 # (short packets: 256-bit windows, no gate)

def test_ax25_frames_round_trip_through_noise(ax25_signal):
    p, x, o, pk = ax25_signal
    got, st = receive(p, x, o.Fs, o.sps, o.delay)
    assert np.array_equal(got, np.unpackbits(np.frombuffer(b"".join(pk), np.uint8)).astype(bool))

def test_ax25_any_block_cut(ax25_signal):
    p, x, o, pk = ax25_signal
    got, _ = receive(p, x, o.Fs, o.sps, o.delay, cuts=[1000, 1, 2500, len(x) - 3501])
    assert np.array_equal(got, np.unpackbits(np.frombuffer(b"".join(pk), np.uint8)).astype(bool))

def test_ax25_audio_centre_shifts_the_tones():
    p = rp.profile.load("ax25/afsk1200")
    assert p.rx().ops[0][1].fo == -1700 and p.params == {"sps": 40, "centre": 1700}

# ---- CCSDS telemetry with Reed-Solomon

@pytest.fixture(scope="module")
def ccsds_signal():
    rng = np.random.default_rng(14)
    p = rp.profile.load("ccsds/tm_rs", sync=KNOWN, depth=2)
    frames = rng.integers(0, 2, 3*8*223*2).astype(bool)
    o = p.tx()(rp.Data(frames, Fs=1e6, sps=1, flush=True))
    return p, o, frames

def test_ccsds_builds_the_pairs(ccsds_signal):
    p = ccsds_signal[0]
    assert names(p.rx()) == ["+Known", "/Constellation", "+Delimiter", "/Marker", "/additive/ccsds", "/block/ccsds_rs", "/rs/ccsds_255_223_dual"]
    assert names(p.tx()) == ["*rs/ccsds_255_223_dual", "*block/ccsds_rs", "*additive/ccsds", "*Marker", "*Constellation", "+Shape"]
    assert 3*4*(32 + 8*255*2) <= len(ccsds_signal[1]) < 3*4*(32 + 8*255*2) + 4*64 # (plus the pulse tail, sent whole on the flush)

def test_ccsds_frames_round_trip_inverted_and_offset(ccsds_signal):
    p, o, frames = ccsds_signal
    x = np.concatenate((np.zeros(37, np.complex64), -o.data, np.zeros(100, np.complex64))) # 180 degrees round, part way in
    got, st = receive(p, x, o.Fs, 4, o.delay)
    assert np.array_equal(got, frames) and st.est["asm"]["polarity"] == -1 and st.est["asm"]["frames"] == 3

def test_ccsds_corrects_symbol_errors(ccsds_signal):
    """Up to 16 wrong bytes per codeword: 12 bursts of 2 bytes in each frame of 2 interleaved codewords, one byte to each"""
    rng = np.random.default_rng(15)
    p, o, frames = ccsds_signal
    tx_ops = [op for how, op in p.tx() if how == "*"]
    d = rp.Data(frames, Fs=1e6, sps=1, flush=True)
    for op in tx_ops[:-1]:
        d = d * op
    bits = np.asarray(d.data).copy()
    per = 32 + 8*255*2
    for f in range(3):
        for k in rng.choice(np.arange(32, per - 16, 16), 12, replace=False):
            bits[f*per + k:f*per + k + 16] ^= True
    rx = p.rx(start=Rep.CHANNEL)
    out = rx(rp.Data(bits, flush=True).derive(bits, rp.Category.CODE, rp.dtype.BITS, rep=Rep.CHANNEL))
    assert np.array_equal(hard(out), frames)

def test_ccsds_any_block_cut(ccsds_signal):
    p, o, frames = ccsds_signal
    x = np.concatenate((np.zeros(37, np.complex64), o.data, np.zeros(100, np.complex64)))
    got, _ = receive(p, x, o.Fs, 4, o.delay, cuts=[4000, 3, 9000, len(x) - 13003])
    assert np.array_equal(got, frames)


# ---- CCSDS telemetry, concatenated (convolutional inner code)

@pytest.fixture(scope="module")
def ccsds_concat_signal():
    rng = np.random.default_rng(16)
    p = rp.profile.load("ccsds/tm_concat", sync=KNOWN, depth=1)
    frames = rng.integers(0, 2, 3*8*223).astype(bool)
    o = p.tx()(rp.Data(frames, Fs=1e6, sps=1, flush=True))
    return p, o, frames

def test_ccsds_concat_round_trip_inverted_and_noisy(ccsds_concat_signal):
    p, o, frames = ccsds_concat_signal
    assert names(p.tx())[:5] == ["*rs/ccsds_255_223_dual", "*block/ccsds_rs", "*additive/ccsds", "*Marker", "*k7_171_133/ccsds_1_2"]
    rng = np.random.default_rng(17)
    x = -o.data + (rng.normal(size=len(o.data)) + 1j*rng.normal(size=len(o.data))).astype(np.complex64)*0.25
    x = np.concatenate((np.zeros(37, np.complex64), x, np.zeros(400, np.complex64)))
    got, st = receive(p, x, o.Fs, 4, o.delay)
    assert np.array_equal(got[:len(frames)], frames)
