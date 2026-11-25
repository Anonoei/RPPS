"""Block interleavers of the shipped standards"""
import numpy as np
import pytest

import rpps as rp
from support.signals import bits

# --- block interleavers ---------------------------------------------------------------------------------------

@pytest.mark.parametrize("ncbps,nbpsc", [(48, 1), (96, 2), (192, 4), (288, 6)])
def test_wifi_interleaver(ncbps, nbpsc, rng):
    il = rp.code.load("interleave/wifi_ofdm", ncbps=ncbps, nbpsc=nbpsc).encoder
    s = max(nbpsc//2, 1)
    k = np.arange(ncbps)
    i = (ncbps//16)*(k % 16) + k//16                   # 17-17
    j = s*(i//s) + (i + ncbps - (16*i)//ncbps) % s     # 17-18
    x = bits(rng, ncbps)
    y = il.encode(x)
    assert np.array_equal(y[j], x)
    assert np.array_equal(il.decode(y), x)

def test_p25_tsbk_interleave():
    order = [d for r in range(12) for d in (2*r, 2*r + 1, 26 + 2*r, 27 + 2*r, 50 + 2*r, 51 + 2*r, 74 + 2*r, 75 + 2*r)] + [24, 25]
    il = rp.code.load("interleave/p25_tsbk").encoder
    rx = np.arange(98*2) # Dibits as pairs of positions
    assert np.array_equal(il.decode(rx).reshape(98, 2)[:, 0]//2, order)

@pytest.mark.parametrize("D", [40, 57])
def test_lte_subblock(D):
    P = [1, 17, 9, 25, 5, 21, 13, 29, 3, 19, 11, 27, 7, 23, 15, 31, 0, 16, 8, 24, 4, 20, 12, 28, 2, 18, 10, 26, 6, 22, 14, 30]
    R = -(-D//32)
    y = [None]*(R*32 - D) + list(range(D))
    want = [y[r*32 + P[c]] for c in range(32) for r in range(R)]
    want = [v for v in want if v is not None]
    il = rp.code.load("interleave/lte_subblock", length=D).encoder
    assert list(il.encode(np.arange(D))) == want

def test_block_interleaver_rejects_non_permutation():
    with pytest.raises(ValueError):
        rp.code.types.block_interleaver(rp.logger.Logger(), [0, 1, 1])
