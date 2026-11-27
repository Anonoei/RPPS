"""Quasi-cyclic LDPC codes from their base matrices"""
import numpy as np
import pytest

import rpps as rp
from rpps.base import Mod
from rpps.base.types import Rep
from support.signals import hard

# ---- a quasi-cyclic LDPC code from its base matrix

@pytest.mark.parametrize("rate,k", [("1_2", 324), ("2_3", 432), ("3_4", 486), ("5_6", 540)])
def test_ieee80211n_ldpc_from_shifts(rate, k, rng):
    c = rp.code.load(f"block/ldpc/ieee80211n_648_{rate}")
    assert c.encoder.num == k and c.encoder.den == 648
    msg = rng.integers(0, 2, 2*k).astype(bool)
    enc = rp.Data(msg) * c
    assert len(enc) == 2*648
    assert not c.decoder.syndrome(np.asarray(enc.data).reshape(2, 648)).any() # every check holds
    llr = (1 - 2*np.asarray(enc.data).astype(np.float32))*2.0 + rng.normal(size=len(enc)).astype(np.float32)*0.8
    dec = enc.derive(llr, rp.Category.SYMBOL, rp.dtype.LLR, Mod, rep=Rep.CHANNEL) / c
    assert np.array_equal(hard(dec), msg) and (dec.errors >= 0).all()

def test_qc_checks_lift_shifts():
    from rpps.code.types.ldpc import qc_checks
    n, checks = qc_checks([[0, -1, 1], [2, 0, -1]], 3)
    assert n == 9 and len(checks) == 6
    assert checks[0].tolist() == [0, 7] and checks[1].tolist() == [1, 8] and checks[2].tolist() == [2, 6] # shift 1: row r -> column (r + 1) % 3
    assert checks[3].tolist() == [2, 3] # shift 2 of block 0: row 0 -> column 2
