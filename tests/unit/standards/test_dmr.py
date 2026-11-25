"""DMR block codes, BPTC and CRC masks against ETSI TS 102 361-1"""
import numpy as np
import pytest

import rpps as rp
from support.signals import bits

# --- product codes, DMR ---------------------------------------------------------------------------------------

# ETSI TS 102 361-1 Table B.11: Golay (20,8) generator matrix, parity part
GOLAY_20_8_P = ["001111011010", "110110011001", "011011001101", "001101100111",
                "110111000110", "101010010111", "100100111110", "100011101011"]

# Table B.18: RS(12,9) parity bytes of each unit message row
RS_12_9 = [(0x1C, 0xBC, 0xFD), (0x89, 0x31, 0x08), (0xAD, 0x41, 0x36), (0x7D, 0x71, 0x16), (0xF3, 0xA6, 0x3A),
           (0x08, 0x83, 0x7B), (0x3F, 0x6F, 0x02), (0x6C, 0x0D, 0xA7), (0x0E, 0x38, 0x40)]

def test_dmr_golay_20_8_is_table_b11():
    g = rp.code.load("block/golay/20_8").encoder.generator.astype(int)
    assert ["".join(map(str, row[8:])) for row in g] == GOLAY_20_8_P
    assert rp.code.load("block/golay/20_8").decoder.t == 3

def test_dmr_rs_12_9_is_table_b18():
    rs = rp.code.load("block/rs/dmr_12_9").encoder
    for r, par in enumerate(RS_12_9):
        assert tuple(rs.encode(np.eye(9, dtype=int)[r], symbols=True)[9:]) == par

@pytest.mark.parametrize("ref,n,k,t", [("hamming/dmr_15_11", 15, 11, 1), ("hamming/dmr_13_9", 13, 9, 1), ("hamming/dmr_16_11", 16, 11, 1),
                                       ("hamming/dmr_17_12", 17, 12, 1), ("hamming/dmr_7_4", 7, 4, 1), ("qr/16_7", 16, 7, 2)])
def test_dmr_block_codes(ref, n, k, t, rng):
    code = rp.code.load(f"block/{ref}", decoder="syndrome")
    assert (code.encoder.num, code.encoder.den, code.decoder.t) == (k, n, t)
    msg = bits(rng, k*30)
    cw = code.encoder.encode(msg).reshape(-1, n).copy()
    for w in cw:
        w[rng.permutation(n)[:t]] ^= True
    assert np.array_equal(code.decoder.decode(cw.reshape(-1)), msg)

def test_dmr_bptc_layout():
    """Table B.3: transmit index -> bit. Matrix index of a bit: R(3) = 0, then row by row (15 per row) from 1"""
    bptc = rp.code.load("block/product/dmr_bptc").encoder
    il = rp.code.load("interleave/block/dmr_bptc").encoder
    for i_bit, tx_index in ((95, 136), (77, 2), (5, 70), (1, 10)): # I(95) at TX index 136, I(77) at 2, I(5) at 70, I(1) at 10
        msg = np.zeros(96, bool)
        msg[95 - i_bit] = True                                  # I(95) is sent first in the message
        tx = il.encode(bptc.encode(msg))
        data_positions = [p for p in range(196) if tx[p]]
        assert tx_index in data_positions

def test_dmr_bptc_corrects(rng):
    bptc = rp.code.load("block/product/dmr_bptc")
    il = rp.code.load("interleave/block/dmr_bptc").encoder
    msg = bits(rng, 96)
    tx = il.encode(bptc.encoder.encode(msg))
    for nerr in (1, 4, 6):
        rx = tx.copy()
        rx[rng.permutation(196)[:nerr]] ^= True
        out, errors = bptc.decoder.decode(il.decode(rx), status=True)
        assert np.array_equal(out, msg) and errors[0] == nerr
    rx = tx.copy()
    rx[:60] ^= True
    assert bptc.decoder.decode(il.decode(rx), status=True)[1][0] == -1

def test_dmr_crc_mask(rng):
    crc = rp.link.load("crc/crc16_ccitt_false").encoder
    sent = crc.encode(bits(rng, 80), mask=0xA5A5) # CSBK
    assert crc.syndrome(sent) == 0xA5A5
