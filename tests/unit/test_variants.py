"""Profile variants (a param's "choices") in blind analysis: rp.profile.variants, identify's match_profile"""
import json

import pytest

import rpps as rp
from rpps import _config
from rpps.analysis.identify import match_profile, _named

VARIANT = {
    "comment": "test: a convolutional code on QPSK or 8PSK",
    "params": {
        "modulation": {"default": "qpsk", "choices": {"qpsk": {"shape": "QPSK"}, "8psk": {"shape": "8PSK"}}},
        "rate": {"default": "1_2", "choices": ["1_2", "3_4"]},
        "payload": {"default": 32, "choices": [32, 64], "blind": False}},
    "steps": [
        {"symbol": "constellation", "shape": {"param": "modulation.shape"}},
        {"code": "conv", "name": {"expr": "'k7_171_133/' + rate"}}]}

@pytest.fixture
def layer(tmp_path):
    f = tmp_path/"profile"/"test"/"variant.json"
    f.parent.mkdir(parents=True)
    f.write_text(json.dumps(VARIANT))
    rp.add_config_path(tmp_path)
    yield
    _config._EXTRA.remove(tmp_path)

def test_variants_defaults_first_blind_false_left_out(layer):
    vs = rp.profile.variants("test/variant")
    assert vs[0] == {"modulation": "qpsk", "rate": "1_2"}
    assert len(vs) == 4 and {(v["modulation"], v["rate"]) for v in vs} == {("qpsk", "1_2"), ("qpsk", "3_4"),
                                                                           ("8psk", "1_2"), ("8psk", "3_4")}
    assert rp.profile.variants("test/variant", limit=3) == vs[:3]
    assert rp.profile.variants("dvbs/qpsk") == [{}] # no choices: the defaults alone

def test_match_profile_finds_a_variant(layer):
    steps = rp.profile.load("test/variant", modulation="8psk", rate="3_4").obj["steps"]
    name, got, params = match_profile(steps)
    assert name == "test/variant" and params == {"modulation": "8psk", "rate": "3_4"} and got == steps
    assert _named(name, params) == "test/variant (modulation=8psk, rate=3_4)"

def test_match_profile_without_a_match():
    assert match_profile([{"symbol": "constellation", "shape": "64QAM"}, {"code": "conv", "name": "k7_171_133/7_8"}]) == \
        (None, None, None)
