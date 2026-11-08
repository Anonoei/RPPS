"""State-machine trellis codes, turbo code, LTE rate matching, product codes; DMR codes against ETSI TS 102 361-1"""
import numpy as np
import pytest

import rpps as rp
from conftest import bits

# --- fsm ------------------------------------------------------------------------------------------------------

def test_p25_half_rate(rng):
    code = rp.coding.load("block", "trellis/p25_1_2")
    msg = bits(rng, 96)
    cw = code._enc.encode(msg)
    assert len(cw) == 196 # 48 dibits + the flush dibit, 4 bits each
    bad = cw.copy()
    bad[rng.permutation(196)[:8]] ^= True
    assert np.array_equal(code._dec.decode(bad), msg) and code.errors[0] == 8

def test_p25_half_rate_soft(rng):
    code = rp.coding.load("block", "trellis/p25_1_2")
    good = 0
    for _ in range(20):
        msg = bits(rng, 96)
        llr = 2*((1 - 2.0*code._enc.encode(msg)) + rng.normal(0, 0.5, 196))/0.25
        good += np.array_equal(code._dec.decode(llr), msg)
    assert good >= 19

def test_atsc_trellis_levels(rng):
    code = rp.coding.load("block", "trellis/atsc_8vsb")
    enc, dec = code._enc, code._dec
    msg = bits(rng, 2*2000)
    lv = enc.modulate(enc.encode(msg))
    assert set(np.unique(lv)) <= set(range(-7, 8, 2))
    assert np.array_equal(dec.decode(lv + rng.normal(0, 0.5, len(lv))), msg)
    # Batches: independent runs from any start state
    many = np.stack([enc.modulate(enc.encode(m)) for m in msg.reshape(4, -1)])
    assert np.array_equal(dec.decode_batch(many).reshape(-1), msg)

def test_fsm_rejects_bad_tables():
    with pytest.raises(ValueError):
        rp.coding.types.fsm(rp.logger.Logger(), 1, 2, [[0, 0], [0, 0]], [[0, 3], [1, 2]]) # state 1 unreachable
    with pytest.raises(ValueError):
        rp.coding.types.fsm(rp.logger.Logger(), 1, 2, [[0, 1], [0, 1]], [[0, 4], [1, 2]]) # word wider than n

# --- turbo, rate matching -------------------------------------------------------------------------------------

def test_turbo_tail_terminates(rng):
    c = rp.coding.types.rsc("0o13", "0o15", 4)
    x = rng.integers(0, 2, 101).astype(np.uint8)
    _, tx, _ = c.encode(x)
    s = 0
    for b in list(x) + list(tx):
        s = c.next[s, b]
    assert s == 0

def test_turbo_through_rate_matching(rng):
    K, E = 248, 900
    t = rp.coding.load("block", "turbo/lte", K=K)._enc
    good = 0
    for rv in range(4):
        rm = rp.coding.load("block", "ratematch/lte_turbo", K=K, E=E, rv=rv)._enc
        for _ in range(3):
            c = bits(rng, K)
            tx = rm.encode(t.encode(c))
            llr = 2*((1 - 2.0*tx) + rng.normal(0, 0.9, E))/0.81
            good += np.array_equal(t.decode(rm.decode(llr)), c)
    assert good >= 11

def test_turbo_qpp_needs_a_listed_size():
    with pytest.raises(ValueError):
        rp.coding.load("block", "turbo/lte", K=41)

def test_conv_rate_matching_repeats(rng):
    D, E = 40, 1920
    rm = rp.coding.load("block", "ratematch/lte_conv", D=D, E=E)._enc
    x = bits(rng, 3*D)
    y = rm.encode(x)
    assert len(y) == E
    sums = rm.decode(1.0 - 2.0*y)
    assert np.allclose(np.abs(sums), E/(3*D)) # Each coded bit repeated 16 times
    assert np.array_equal(sums < 0, x)

def test_turbo_rate_matching_punctures(rng):
    K = 40
    rm = rp.coding.load("block", "ratematch/lte_turbo", K=K, E=100, rv=0)._enc
    sums = rm.decode(np.ones(100))
    assert (sums > 0).sum() == 100 and len(sums) == 3*(K + 4) # 100 distinct positions, the rest erased

# --- product codes, DMR ---------------------------------------------------------------------------------------

# ETSI TS 102 361-1 Table B.11: Golay (20,8) generator matrix, parity part
GOLAY_20_8_P = ["001111011010", "110110011001", "011011001101", "001101100111",
                "110111000110", "101010010111", "100100111110", "100011101011"]
# Table B.18: RS(12,9) parity bytes of each unit message row
RS_12_9 = [(0x1C, 0xBC, 0xFD), (0x89, 0x31, 0x08), (0xAD, 0x41, 0x36), (0x7D, 0x71, 0x16), (0xF3, 0xA6, 0x3A),
           (0x08, 0x83, 0x7B), (0x3F, 0x6F, 0x02), (0x6C, 0x0D, 0xA7), (0x0E, 0x38, 0x40)]

def test_dmr_golay_20_8_is_table_b11():
    g = rp.coding.load("block", "golay/20_8")._enc.generator.astype(int)
    assert ["".join(map(str, row[8:])) for row in g] == GOLAY_20_8_P
    assert rp.coding.load("block", "golay/20_8")._dec.t == 3

def test_dmr_rs_12_9_is_table_b18():
    rs = rp.coding.load("block", "rs/dmr_12_9")._enc
    for r, par in enumerate(RS_12_9):
        assert tuple(rs.encode(np.eye(9, dtype=int)[r], symbols=True)[9:]) == par

@pytest.mark.parametrize("ref,n,k,t", [("hamming/dmr_15_11", 15, 11, 1), ("hamming/dmr_13_9", 13, 9, 1), ("hamming/dmr_16_11", 16, 11, 1),
                                       ("hamming/dmr_17_12", 17, 12, 1), ("hamming/dmr_7_4", 7, 4, 1), ("qr/16_7", 16, 7, 2)])
def test_dmr_block_codes(ref, n, k, t, rng):
    code = rp.coding.load("block", ref, decoder="syndrome")
    assert (code._enc.num, code._enc.den, code._dec.t) == (k, n, t)
    msg = bits(rng, k*30)
    cw = code._enc.encode(msg).reshape(-1, n).copy()
    for w in cw:
        w[rng.permutation(n)[:t]] ^= True
    assert np.array_equal(code._dec.decode(cw.reshape(-1)), msg)

def test_dmr_bptc_layout():
    """Table B.3: transmit index -> bit. Matrix index of a bit: R(3) = 0, then row by row (15 per row) from 1"""
    bptc = rp.coding.load("block", "product/dmr_bptc")._enc
    il = rp.coding.load("interleave", "block/dmr_bptc")._enc
    for i_bit, tx_index in ((95, 136), (77, 2), (5, 70), (1, 10)): # I(95) at TX index 136, I(77) at 2, I(5) at 70, I(1) at 10
        msg = np.zeros(96, bool)
        msg[95 - i_bit] = True                                  # I(95) is sent first in the message
        tx = il.encode(bptc.encode(msg))
        data_positions = [p for p in range(196) if tx[p]]
        assert tx_index in data_positions

def test_dmr_bptc_corrects(rng):
    bptc = rp.coding.load("block", "product/dmr_bptc")
    il = rp.coding.load("interleave", "block/dmr_bptc")._enc
    msg = bits(rng, 96)
    tx = il.encode(bptc._enc.encode(msg))
    for nerr in (1, 4, 6):
        rx = tx.copy()
        rx[rng.permutation(196)[:nerr]] ^= True
        assert np.array_equal(bptc._dec.decode(il.decode(rx)), msg) and bptc.errors[0] == nerr
    rx = tx.copy()
    rx[:60] ^= True
    bptc._dec.decode(il.decode(rx))
    assert bptc.errors[0] == -1

def test_dmr_crc_mask(rng):
    crc = rp.encap.load("crc", "dmr_ccitt")._enc
    sent = crc.encode(bits(rng, 80), mask=0xA5A5) # CSBK
    assert crc.syndrome(sent) == 0xA5A5
