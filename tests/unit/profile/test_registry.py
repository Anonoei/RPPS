"""Profile steps come from a registry: what ships is registered there, and so can anything else"""
import json

import numpy as np
import pytest

import rpps as rp
from rpps.base import Stage
from rpps.errors import ConfigError
from rpps.profile import steps


class Gain(Stage):
    """A user's own stage"""
    def __init__(self, k=1.0):
        self.k = k
    def step(self, x, st, meta):
        return x*self.k


def test_every_category_and_kind_is_registered():
    kinds = steps.kinds()
    assert {"signal", "sync", "wave", "symbol", "code", "link"} <= set(kinds)
    assert {"filter", "decimate", "class", "dc_block", "deemphasis", "shift"} <= set(kinds["signal"])
    assert {"timing", "carrier", "equalize", "squelch"} <= set(kinds["sync"])
    assert {"oerder_meyr", "mat", "known", "fixed", "mm", "gardner", "tll"} <= set(steps.methods("timing"))

def test_unknown_kind_lists_what_there_is():
    with pytest.raises(ConfigError, match="have .*decimate"):
        steps.build({"signal": "nope"}, rp.base.types.Rep.SAMPLES, steps.context())

def test_a_registered_kind_builds(monkeypatch):
    calls = []
    def gain(kind, spec, rep, ctx):
        calls.append(spec)
        return [(steps.STAGE, Gain(spec.get("k", 2.0)))], [], rep
    monkeypatch.setitem(steps.REGISTRY["signal"], "gain", gain)
    st = steps.build({"signal": "gain", "k": 3.0}, rp.base.types.Rep.SAMPLES, steps.context())
    assert isinstance(st.rx[0][1], Gain) and st.rx[0][1].k == 3.0 and calls

def test_register_step_rejects_a_taken_name():
    with pytest.raises(ValueError):
        steps.register_step("signal", "filter")(lambda *a: None)

def test_generic_stage_step_runs_a_user_class(tmp_path, config_layer):
    """A profile in a user's config directory names a class by its module path"""
    prof = {"comment": "gain then DC block",
            "steps": [{"signal": "class", "class": "test_registry.Gain", "args": {"k": 2.0}},
                      {"signal": "dc_block"}]}
    d = tmp_path / "profile" / "mine"
    d.mkdir(parents=True)
    (d / "gain.json").write_text(json.dumps(prof))
    config_layer(tmp_path)
    p = rp.profile.load("mine/gain")
    rx = p.rx()
    x = np.ones(64, np.complex64)
    y = np.asarray(rx(rp.Data(x, Fs=1e3)).data)
    assert len(y) == 64 and abs(y[0] - 2.0) < 1e-6

def test_stage_step_needs_a_stage():
    with pytest.raises(ConfigError, match="not a Stage"):
        steps.build({"signal": "class", "class": "json.JSONDecoder"}, rp.base.types.Rep.SAMPLES, steps.context())

def test_tll_timing_method_from_a_profile_step():
    st = steps.build({"sync": "timing", "method": "tll", "bw": 0.01, "sps": 4}, rp.base.types.Rep.SAMPLES, steps.context())
    assert isinstance(st.rx[0][1], rp.sync.loops.TLL)

def test_load_any_definition_by_path():
    assert rp.load("fec/block/rs/204_188").d_den == 204
    assert rp.load("crc/crc32_mpeg2") is not None
    assert rp.load("profile/ieee802154/2450", sps=8).params == {"sps": 8}
    with pytest.raises(ConfigError, match="no resource"):
        rp.load("nope/x")
    with pytest.raises(ConfigError, match="name a definition"):
        rp.load("crc")

class Swap(rp.base.rpps.Pipe):
    """A link operator of a user's own: each pair of bits swapped (its own inverse)"""
    CATEGORY = rp.base.types.Category.LINK
    def __init__(self, n=2):
        self.n = n
    def takes(self, encode):
        return None
    def _swap(self, data):
        x = np.asarray(data.data).astype(bool)
        whole = len(x)//self.n*self.n
        y = np.concatenate((x[:whole].reshape(-1, self.n)[:, ::-1].reshape(-1), x[whole:]))
        return data.derive(y, self.CATEGORY, rp.dtype.BITS, rp.base.Digital, data.ctx, rep=rp.base.types.Rep.PACKETS)
    encode = decode = _swap

def test_a_pipe_class_step_in_a_transform_category():
    st = steps.build({"link": "class", "class": "test_registry.Swap", "args": {"n": 2}}, rp.base.types.Rep.DATA, steps.context())
    assert [how for how, _ in st.rx] == ["/"] and [how for how, _ in st.tx] == ["*"] and st.gives == rp.base.types.Rep.PACKETS
    bits = np.array([1, 0, 0, 0, 1, 1], bool)
    out = rp.Data(bits) * st.tx[0][1]
    assert np.array_equal(np.asarray(out.data), [0, 1, 0, 0, 1, 1])
    assert np.array_equal(np.asarray((out / st.rx[0][1]).data), bits)

def test_a_class_step_in_the_wrong_category():
    with pytest.raises(ConfigError, match="does LINK, not code"):
        steps.build({"code": "class", "class": "test_registry.Swap"}, rp.base.types.Rep.DATA, steps.context())
    with pytest.raises(ConfigError, match="is a Stage, not a Pipe"):
        steps.build({"link": "class", "class": "test_registry.Gain"}, rp.base.types.Rep.DATA, steps.context())
