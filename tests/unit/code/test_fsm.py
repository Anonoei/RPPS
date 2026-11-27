"""Trellis codes as state machines: P25 and ATSC"""
import numpy as np
import pytest

import rpps as rp
from support.signals import bits

# --- fsm ------------------------------------------------------------------------------------------------------

def test_p25_half_rate(rng):
    code = rp.code.load("block/trellis/p25_1_2")
    msg = bits(rng, 96)
    cw = code.encoder.encode(msg)
    assert len(cw) == 196 # 48 dibits + the flush dibit, 4 bits each
    bad = cw.copy()
    bad[rng.permutation(196)[:8]] ^= True
    out, errors = code.decoder.decode(bad, status=True)
    assert np.array_equal(out, msg) and errors[0] == 8

def test_paths_give_what_the_decoder_decided_was_sent(rng):
    code = rp.code.load("block/trellis/p25_1_2")
    dec = code.decoder
    msg = bits(rng, 96*3).reshape(3, 96)
    cw = np.array([code.encoder.encode(m) for m in msg])
    bad = cw.copy()
    bad[:, 10] ^= True
    out, words = dec.paths(bad)
    assert np.array_equal(out, dec.decode_batch(bad)) and np.array_equal(out, msg)
    sent = ((words[..., None] >> np.arange(dec.n - 1, -1, -1)) & 1).astype(bool).reshape(3, -1)
    assert np.array_equal(sent, cw) # The flipped bit is not in the path: the path is the codeword

def test_p25_half_rate_soft(rng):
    code = rp.code.load("block/trellis/p25_1_2")
    good = 0
    for _ in range(20):
        msg = bits(rng, 96)
        llr = 2*((1 - 2.0*code.encoder.encode(msg)) + rng.normal(0, 0.5, 196))/0.25
        good += np.array_equal(code.decoder.decode(llr), msg)
    assert good >= 19

def test_atsc_trellis_levels(rng):
    code = rp.code.load("block/trellis/atsc_8vsb")
    enc, dec = code.encoder, code.decoder
    msg = bits(rng, 2*2000)
    lv = enc.modulate(enc.encode(msg))
    assert set(np.unique(lv)) <= set(range(-7, 8, 2))
    assert np.array_equal(dec.decode(lv + rng.normal(0, 0.5, len(lv))), msg)
    # Batches: independent runs from any start state
    many = np.stack([enc.modulate(enc.encode(m)) for m in msg.reshape(4, -1)])
    assert np.array_equal(dec.decode_batch(many).reshape(-1), msg)

def test_fsm_rejects_bad_tables():
    with pytest.raises(ValueError):
        rp.code.types.fsm(rp.logger.Logger(), 1, 2, [[0, 0], [0, 0]], [[0, 3], [1, 2]]) # state 1 unreachable
    with pytest.raises(ValueError):
        rp.code.types.fsm(rp.logger.Logger(), 1, 2, [[0, 1], [0, 1]], [[0, 4], [1, 2]]) # word wider than n
