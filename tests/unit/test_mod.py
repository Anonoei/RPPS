import numpy as np
import pytest

import rpps as rp

NAMES = [n for names in rp.mod.ls().values() for n in names]

@pytest.mark.parametrize("name", NAMES)
def test_constellation_normalised(name):
    pts = np.asarray(rp.mod.load(name).points.arr)
    assert len(pts) == 2**int(np.log2(len(pts)))
    assert np.mean(np.abs(pts)**2) == pytest.approx(1.0, rel=0.05)

@pytest.mark.parametrize("name", NAMES)
def test_mod_demod_round_trip(name, rng):
    m = rp.mod.load(name)
    m.set_mapping(m.get_maps()[0])
    msg = rp.Data(bytes(rng.integers(0, 256, 24).astype(np.uint8))).as_bits() # 192 bits: whole symbols for 1, 2, 3 bits/symbol
    out = (msg * m) / m
    assert out.as_bytes().hex == msg.as_bytes().hex
