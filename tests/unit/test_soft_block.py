"""Blind block codes from soft values, where hard bits are too noisy for the rank (rpps.analysis.code.block.linear)"""
import numpy as np

from rpps.analysis.code.block import linear as LIN
from rpps.code.types.bchgf import bchgf
from rpps.logger import Logger

def bch_llr(seed, ebn0, W=1500):
    """BCH(63,51) words through BPSK and AWGN at Eb/N0 (dB): (hard bits, LLRs)"""
    code = bchgf(Logger().Child("t", 0), 63, 51, 6, 0x43, 2)
    rng = np.random.default_rng(seed)
    C = np.asarray(code.encode(rng.integers(0, 2, 51*W).astype(bool)), np.uint8)
    sig = np.sqrt(1/(2*10**(ebn0/10)*51/63))
    llr = 2*((1 - 2.0*C) + sig*rng.standard_normal(C.shape))/sig**2
    return (llr < 0).astype(np.uint8), llr

def test_soft_dual_where_the_hard_one_fails():
    """5.5 dB (about 1 bit in 100 wrong, half the words hit). Measured at 5 dB over 10 seeds: soft 6, hard 0 whole duals"""
    hard, llr = bch_llr(1, 5.5)
    assert LIN.recover_dual(hard, 63, trials=20)["k"] != 51
    r = LIN.soft_dual(llr, 63, trials=20)
    assert r["ok"] and r["k"] == 51
    assert np.all(LIN.soft_score(r["H"], llr.reshape(-1, 63)) > 8)

def test_bch_named_blind_from_soft_values():
    """Measured over 10 seeds: named at 5.5 dB 10 times, at 5 dB 8; hard bits alone stop above 6 dB"""
    hard, llr = bch_llr(2, 5.5)
    assert LIN.solve_block(hard) == []
    found = LIN.solve_block(hard, llr=llr, soft_length=True)
    assert found and "BCH(63,51)" in found[0][0].name and found[0][0].params["soft"]

def test_no_code_in_random_soft_values():
    """Measured: 0 of 10 sets of random bits with LLRs give a code (also with the soft length search)"""
    for s in range(3):
        rng = np.random.default_rng(300 + s)
        b = rng.integers(0, 2, 63*1500).astype(np.uint8)
        assert LIN.solve_block(b, llr=(1 - 2.0*b)*np.abs(rng.normal(2, 1.5, len(b)))) == []
