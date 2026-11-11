import numpy as np
import pytest

import rpps as rp

NAMES = [n for cat, names in rp.symbol.ls().items() if cat.startswith("constellation") for n in names] # every constellation definition
assert NAMES, "no constellation definitions found" # an empty list would silently skip the tests below

@pytest.mark.parametrize("name", NAMES)
def test_constellation_normalised(name):
    pts = np.asarray(rp.symbol.load(name).points)
    assert len(pts) == 2**int(np.log2(len(pts)))
    assert np.mean(np.abs(pts)**2) == pytest.approx(1.0, rel=0.05)

@pytest.mark.parametrize("name", NAMES)
def test_mod_demod_round_trip(name, rng):
    m = rp.symbol.load(name)
    m.set_mapping(m.maps[0])
    msg = rp.Data(bytes(rng.integers(0, 256, 24).astype(np.uint8))).as_bits() # 192 bits: whole symbols for 1, 2, 3 bits/symbol
    out = (msg * m) / m
    assert out.as_bytes().hex == msg.as_bytes().hex
