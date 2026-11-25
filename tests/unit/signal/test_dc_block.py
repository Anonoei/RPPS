"""DC block"""
import numpy as np

import rpps as rp

def test_dc_block_removes_the_offset():
    x = (0.7 + 0.1*np.exp(2j*np.pi*0.1*np.arange(20000))).astype(np.complex64)
    out = (rp.Data(x) + rp.signal.DCBlock()).data
    assert abs(np.mean(out[10000:])) < 1e-2
