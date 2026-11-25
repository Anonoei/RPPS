"""Sync words, markers and delimiters"""
import numpy as np
import pytest

import rpps as rp
from rpps.base.types import Rep
from rpps.errors import ConfigError
from support.signals import hard, as_data_bits, as_llr

# ---- words and markers

def test_word_sequence():
    w = rp.sequence.load("ccsds_asm")
    assert w.n == 32 and w.value == 0x1ACFFC1D and list(w.bits[:8].astype(int)) == [0, 0, 0, 1, 1, 0, 1, 0]
    ble = rp.sequence.load("ble_adv_aa")
    assert ble.order == "lsb" and ble.n == 40 and list(ble.bits[:8].astype(int)) == [0, 1, 0, 1, 0, 1, 0, 1]
    assert rp.sequence.load("mode_s_preamble").value == 0xA140
    with pytest.raises(ConfigError):
        rp.sequence.word.word_bits("0x1FF", 8)

def test_marker_fixed_length_round_trip_and_errors(rng):
    frames = rng.integers(0, 2, 3*64).astype(bool)
    M = rp.code.Marker("0x1ACFFC1D", 32, length=64)
    coded = rp.Data(frames, flush=True) * M
    assert len(coded) == 3*96 and coded.REP == Rep.CHANNEL
    assert M.pos_rate(False) == 96/64 and M.pos_rate(True) == 64/96
    bits = np.asarray(coded.data).copy()
    bits[96 + 3] ^= True # one marker bit wrong in the second frame
    back = as_data_bits(bits, Rep.CHANNEL, flush=True) / M
    assert np.array_equal(hard(back), frames) and list(back.errors) == [0, 1, 0]
    soft = as_llr(coded) / M
    assert soft.DT == rp.dtype.LLR and np.array_equal(hard(soft), frames)

def test_delimiter_keeps_the_word_fixes_polarity_and_the_marker_strips_it(rng):
    w = rp.sequence.load("ccsds_asm")
    frames = rng.integers(0, 2, 4*64).astype(bool)
    M = rp.code.Marker(w.bits, length=64)
    coded = np.asarray((rp.Data(frames, flush=True) * M).data)
    stream = np.concatenate((rng.integers(0, 2, 20).astype(bool), ~coded, rng.integers(0, 2, 9).astype(bool))) # inverted
    D = rp.sync.Delimiter(w, length=64, keep=True, polarity=True)
    for cuts in ([len(stream)], [13, 100, 7, len(stream) - 120]):
        outs, st, i = [], None, 0
        for k, c in enumerate(cuts):
            d = as_data_bits(stream[i:i + c], Rep.CHANNEL, stream=st, flush=k == len(cuts) - 1)
            st, i = d.stream, i + c
            outs.append(hard(d + D))
        aligned = np.concatenate(outs)
        assert np.array_equal(aligned, coded) and st.est["delimiter"]["polarity"] == -1 and st.est["delimiter"]["frames"] == 4
    back = as_data_bits(aligned, Rep.CHANNEL, flush=True) / M
    assert np.array_equal(hard(back), frames) and not any(back.errors)

def test_delimiter_without_keep_gives_the_bits_after_the_word():
    bits = np.array([1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1], bool) # the word 0100 at index 2, then 11; no other exact match
    out = as_data_bits(bits, Rep.CHANNEL, flush=True) + rp.sync.Delimiter(0b0100, 4, length=2)
    assert list(hard(out).astype(int)) == [1, 1]
    out = as_data_bits(bits, Rep.CHANNEL, flush=True) + rp.sync.Delimiter(0b0100, 4, length=2, errors=1)
    assert list(hard(out).astype(int)) == [1, 1, 1, 1] # with one error allowed, 0101 at index 8 too (then 11)

@pytest.mark.parametrize("order", ["msb", "lsb"])
def test_a_word_of_any_width(order, rng):
    """A 100-byte unique word: as a hex string, its bits and value round trip (no 64-bit limit)"""
    raw = rng.integers(0, 256, 100, dtype=np.uint8)
    w = rp.sequence.word.Word("uw", "0x" + raw.tobytes().hex(), 800, order)
    msb = np.unpackbits(raw).astype(bool)
    assert w.n == 800 and w.value == int(raw.tobytes().hex(), 16)
    assert np.array_equal(w.bits, msb[::-1] if order == "lsb" else msb)
    with pytest.raises(ConfigError):
        rp.sequence.word.word_bits("0x" + "ff"*101, 800)

def test_a_long_word_is_delimited_with_errors_across_any_block_cut(rng):
    """800 bits with 40 errors in every copy: matched by FFT correlation, exactly where the window-by-window count would,
    the inverted frames put right, whatever the block cut and as LLRs too"""
    w = rng.integers(0, 2, 800).astype(bool)
    frames = rng.integers(0, 2, 5*3000).astype(bool)
    M = rp.code.Marker(w, length=3000)
    coded = np.asarray((rp.Data(frames, flush=True) * M).data).copy()
    for k in range(5): # 40 bit errors in each word, the third frame inverted
        coded[k*3800 + rng.choice(800, 40, replace=False)] ^= True
    coded[2*3800:3*3800] ^= True
    stream = np.concatenate((rng.integers(0, 2, 333).astype(bool), coded))
    allowed = rp.sync.word_errors(800, 1e-9, 3800, polarity=True)
    assert 250 < allowed < 320
    D = rp.sync.Delimiter(w, length=3000, keep=True, polarity=True, errors=allowed)
    hard_bits = stream
    starts = [s for s, _ in D._starts(hard_bits, 0)]
    win = np.lib.stride_tricks.sliding_window_view(hard_bits, 800)
    bad = np.count_nonzero(win != w[None, :], axis=1)
    assert starts == sorted(set(np.nonzero(bad <= allowed)[0]) | set(np.nonzero(800 - bad <= allowed)[0]))
    assert starts == [333 + k*3800 for k in range(5)]
    for cuts, soft in (([len(stream)], False), ([1000, 7, 5000, len(stream)], True)):
        outs, st, i = [], None, 0
        for k, c in enumerate(cuts):
            d = as_data_bits(stream[i:i + c], Rep.CHANNEL, stream=st, flush=i + c >= len(stream))
            st, i = d.stream, i + c
            outs.append(hard((as_llr(d) if soft else d) + D))
            if i >= len(stream):
                break
        back = as_data_bits(np.concatenate(outs), Rep.CHANNEL, flush=True) / M
        assert np.array_equal(hard(back), frames) and list(back.errors) == [40]*5

def test_word_errors_bounds_random_matches(rng):
    """The count word_errors allows keeps random windows under the false-alarm rate, and one more would not"""
    e = rp.sync.word_errors(40, 1e-2)
    within = rng.integers(0, 2, (200000, 40)).sum(axis=1) # (errors against the all-zeros word)
    assert np.mean(within <= e) < 1e-2 < np.mean(within <= e + 1)
    assert rp.sync.word_errors(32, 1e-6, 8160) == -1 and rp.sync.word_errors(32, 1e-3, 8160) == 1

@pytest.mark.parametrize("turn", range(4))
def test_frame_sync_finds_a_bit_word_under_any_rotation(turn, rng):
    """A profile's frame step made from a 100-byte word, through its own QPSK mapping: found coherently, so a carrier
    left 90, 180 or 270 degrees off is put right by the word, and the marker after the mapping strips it"""
    uw = "0x" + rng.integers(0, 256, 100, dtype=np.uint8).tobytes().hex()
    p = rp._config.load_obj("profile", "test/uw", {"base": "profile", "steps": [
        {"sync": "frame", "id": "uw", "name": "uw", "word": uw, "bits": 800, "threshold": 0.3},
        {"symbol": "constellation", "name": "qpsk"},
        {"code": "marker", "word": uw, "bits": 800, "length": 4000}]})
    pay = rng.integers(0, 2, 4*4000).astype(bool)
    sy = np.asarray(p.tx()(rp.Data(pay, flush=True)).data)
    lead = (rng.normal(size=777) + 1j*rng.normal(size=777))/np.sqrt(2)
    y = np.concatenate((lead, sy))*np.exp(1j*(np.pi/2*turn + 0.3))
    y = y + 0.2*(rng.normal(size=len(y)) + 1j*rng.normal(size=len(y)))
    outs = list(rp.source.Memory(y.astype(np.complex64), sps=1)(1234) + p.rx())
    got = np.concatenate([hard(o) for o in outs])
    assert len(got) == len(pay) and np.mean(got != pay) < 1e-3

def test_frame_step_takes_a_header_or_a_word():
    for steps in ([{"sync": "frame"}], [{"sync": "frame", "header": "dvbs2_pl", "word": "0x1", "bits": 1}]):
        with pytest.raises(ConfigError, match="one of the two"):
            rp._config.load_obj("profile", "test/bad", {"base": "profile", "steps": steps}).rx()

def test_delimiter_frames_by_control_strip(rng):
    """Frames of no fixed length: a profile's frame control tells the marker where each starts"""
    w = rp.sequence.load("mode_s_preamble")
    M = rp.code.Marker(w.bits)
    assert M.pos_rate(True) == 1.0
    a, b = rng.integers(0, 2, 40).astype(bool), rng.integers(0, 2, 24).astype(bool)
    ctx = None
    outs = []
    for part in (a, b):
        M.control(frame=1)
        d = rp.Data(part) if ctx is None else rp.Data(part, ctx=ctx)
        o = d * M
        ctx = o.ctx
        outs.append(hard(o))
    assert len(outs[0]) == 56 and len(outs[1]) == 40 and np.array_equal(outs[0][:16], w.bits) and np.array_equal(outs[1][16:], b)
    M2 = rp.code.Marker(w.bits)
    M2.control(frame=0)
    back = as_data_bits(outs[0], Rep.CHANNEL) / M2
    assert np.array_equal(hard(back), a) and list(back.errors) == [0]
