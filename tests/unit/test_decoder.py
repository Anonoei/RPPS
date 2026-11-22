"""What a profile needs before it runs (rp.profile.info, catalogue), and rp.profile.Decoder: a selected signal's samples
to a profile's messages"""
import json

import numpy as np
import pytest

import rpps as rp
from rpps.errors import ConfigError

def test_every_profile_says_what_it_needs():
    from rpps import _config
    cat = rp.profile.catalogue()
    assert {i.name for i in cat} == {f"{f}/{m}" for f, ms in rp.profile.ls().items() for m in ms}
    shipped = [i for i in cat if _config.PATH_CONFIG in _config.find("profile", i.name).parents] # (not a test's own)
    assert len(shipped) >= 19
    for i in shipped:
        r = i.requires
        assert r["input"] in ("baseband", "audio") and r["min_samples"] > 0
        if r["baud"] is None: # A rate the standard leaves open: Fs and bw wait for it
            assert r["Fs"] is None and r["bw"] is None
        else:
            assert r["Fs"] >= r["bw"] > 0
        json.dumps(i.to_dict())

def test_requires_follow_the_params_and_baud():
    assert rp.profile.info("lte/downlink", n_rb=6).requires["Fs"] == 1.92e6
    assert rp.profile.info("adsb/mode_s", sps=2).requires["Fs"] == 4e6
    r = rp.profile.info("dvbs2/normal", baud=27.5e6).requires
    assert r["Fs"] == 4*27.5e6 and r["bw"] == pytest.approx(1.35*27.5e6)
    assert rp.profile.info("ble/adv").requires["from_cf"]["channel"]
    with pytest.raises(ValueError, match="rate is set"):
        rp.profile.info("ble/adv", baud=2e6)
    with pytest.raises(ConfigError, match="no params"):
        rp.profile.info("ble/adv", nope=1)
