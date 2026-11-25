"""802.15.4 through its profile"""
import numpy as np
import pytest

import rpps as rp
from rpps.base.types import Rep
from support.signals import hard
from support.profiles import names
from support.dvbs2 import KNOWN

def test_802154_builds_the_documented_pairs():
    p = rp.profile.load("ieee802154/2450", sync=KNOWN)
    assert [f"{s.category}:{s.kind}" for s in p.steps] == ["sync:carrier", "wave:stagger", "wave:shape", "sync:timing", "symbol:spread",
                                                       "code:interleave"]
    # carrier: method none. Known timing takes the matched filter in (it is evaluated at the instants it samples) ...
    assert names(p.rx()) == ["+Stagger", "+Known", "/Spread", "/block/lsb_first"] # octets least significant bit first
    assert str(p.rx().ops[1][1]) == "Known(matched)"
    # ... unless asked not to
    apart = rp.profile.load("ieee802154/2450", sync={"timing": {"method": "known", "fuse": False}})
    assert names(apart.rx()) == ["+Stagger", "+Matched", "+Known", "/Spread", "/block/lsb_first"]
    assert names(p.tx()) == ["*block/lsb_first", "*Spread", "+Shape", "+Stagger"]
    assert str(p.rx().ops[0][1]) == "Stagger(i)" and str(p.tx().ops[-1][1]) == "Stagger(q)"
    assert [s.takes for s in p.steps] == [Rep.SAMPLES, Rep.SAMPLES, Rep.SAMPLES, Rep.SAMPLES, Rep.SYMBOLS, Rep.CHANNEL]

@pytest.mark.parametrize("sps", [2, 4, 8])
def test_802154_loopback(sps):
    rng = np.random.default_rng(sps)
    p = rp.profile.load("ieee802154/2450", sync=KNOWN, sps=sps)
    msg = rng.integers(0, 256, 60).astype(np.uint8)
    wave = p.tx()(rp.Data(msg, Fs=250e3, sps=1))
    assert wave.REP == Rep.SAMPLES and wave.sps == sps
    out = p.rx()(wave)
    assert out.REP == Rep.DATA
    got, bits = hard(out), np.unpackbits(msg).astype(bool)
    assert len(got) >= len(bits) - 8 and np.array_equal(got, bits[:len(got)])
