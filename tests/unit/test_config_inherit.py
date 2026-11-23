"""Definitions that extend others (parents and children), abstract parents, and tags"""
import json

import numpy as np
import pytest

import rpps as rp
from rpps import _config
from rpps.errors import ConfigError

@pytest.fixture
def layer(tmp_path):
    """A user config layer: write(path, obj) puts a definition in it"""
    def write(rel, obj):
        f = tmp_path / f"{rel}.json"
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(obj))
        return f
    rp.add_config_path(tmp_path)
    yield write
    _config._EXTRA.remove(tmp_path)

def test_merge_rules():
    parent = {"base": "x", "comment": "p", "ref": "spec", "tags": ["a"], "n": 1, "d": {"k": 1, "j": 2}, "l": [1, 2],
              "params": {"g": {"default": 1, "comment": "gain"}}, "gone": 5}
    child = {"extends": "p", "comment": "c", "tags": ["b", "a"], "d": {"j": 3}, "l": [9], "params": {"g": 2}, "gone": None}
    out = _config.merge(parent, child)
    assert out["comment"] == "c" and "ref" not in out # comment and ref are each file's own
    assert out["tags"] == ["a", "b"] # added, no repeats
    assert out["d"] == {"k": 1, "j": 3} and out["l"] == [9] and out["n"] == 1 # dicts merge, lists replace
    assert out["params"]["g"] == {"default": 2, "comment": "gain"} # a bare value pins the parent's default
    assert "gone" not in out and "extends" not in out # null deletes

def test_a_child_loads_as_its_parent_changed(layer):
    layer("constellation/t/base", {"base": "constellation", "comment": "QPSK", "tags": ["dvb-s2"], "rings": [{"n": 4}],
                                   "maps": [{"map": [0, 1, 3, 2], "comment": "R000"}]})
    layer("constellation/t/axes", {"base": "constellation", "extends": "t/base", "comment": "on the axes", "rotate": -45})
    a, b = rp.symbol.load("t/base"), rp.symbol.load("t/axes")
    assert np.allclose(b.points, a.points*np.exp(-1j*np.pi/4))
    assert np.array_equal(a.mapping.arr, b.mapping.arr)
    assert _config.file_tags(_config.find("constellation", "t/axes")) == ["dvb-s2"]

def test_extends_is_looked_up_below_the_category_first(layer):
    layer("fec/block/rs/fam", {"base": "block", "type": "rs", "abstract": True, "comment": "family", "m": 8, "poly": 285})
    layer("fec/block/rs/fam_15_11", {"base": "block", "extends": "rs/fam", "comment": "a member", "n": 15, "k": 11})
    c = rp.code.load("block/rs/fam_15_11")
    assert c.encoder.num == 11 and c.encoder.den == 15

def test_abstract_parents_are_hidden_and_refuse_to_load(layer):
    layer("crc/abs_parent", {"base": "crc", "abstract": True, "comment": "half a CRC", "width": 16, "poly": "0x1021"})
    layer("crc/abs_child", {"base": "crc", "extends": "abs_parent", "type": "crc", "comment": "XMODEM", "init": "0x0",
                            "xorout": "0x0", "refin": False, "refout": False, "check": "0x31c3"})
    assert "abs_parent" not in _config.ls("crc")["crc"] and "abs_parent" in _config.ls("crc", all=True)["crc"]
    with pytest.raises(ConfigError, match="abstract"):
        rp.link.load("crc/abs_parent")
    assert rp.link.load("crc/abs_child").encoder.compute(b"123456789") == 0x31c3

def test_loops_and_other_resources_are_refused(layer):
    layer("crc/loop_a", {"base": "crc", "extends": "loop_b", "comment": "a"})
    layer("crc/loop_b", {"base": "crc", "extends": "loop_a", "comment": "b"})
    with pytest.raises(ConfigError, match="loops"):
        _config.read(_config.find("crc", "loop_a"))
    layer("crc/cross", {"base": "crc", "extends": "constellation/psk/qpsk", "comment": "wrong kind"})
    with pytest.raises(ConfigError, match="its own kind"):
        _config.read(_config.find("crc", "cross"))
    layer("crc/orphan", {"base": "crc", "extends": "no_such_parent", "comment": "?"})
    with pytest.raises(ConfigError, match="not a crc definition"):
        _config.read(_config.find("crc", "orphan"))

def test_the_raw_file_keeps_extends(layer):
    f = layer("crc/raw_child", {"base": "crc", "extends": "crc16_xmodem", "comment": "same"})
    assert _config.read_raw(f) == {"base": "crc", "extends": "crc16_xmodem", "comment": "same"}
    assert _config.read(f)["poly"] == _config.read(_config.find("crc", "crc16_xmodem"))["poly"]

def test_every_tag_used_is_declared():
    known = set(_config.tags())
    used = set(_config.tagged(all=True))
    assert used <= known, f"undeclared tags (add them to config/tags.json): {sorted(used - known)}"

def test_tagged_lists_across_resources(layer):
    layer("crc/tagged_x", {"base": "crc", "extends": "crc16_xmodem", "comment": "x", "tags": ["test-tag"]})
    found = _config.tagged("test-tag")
    assert found == {"crc": ["tagged_x"]}
    with pytest.raises(ConfigError, match="No tag"):
        _config.tagged("no-such-tag-anywhere")

def test_save_needs_a_destination(monkeypatch):
    monkeypatch.setattr(_config, "_EXTRA", [])
    with pytest.raises(ConfigError, match="destination"):
        _config.save("crc", "crc", "x", {"base": "crc"})

def test_a_child_patches_single_steps(layer):
    layer("profile/t/base", {"base": "profile", "comment": "parent", "params": {"sps": {"default": 4}},
                             "steps": [{"wave": "shape", "pulse": "rrc/0p35", "sps": {"param": "sps"}}, {"sync": "timing"},
                                       {"sync": "carrier", "order": 2, "constellation": "bpsk"},
                                       {"symbol": "constellation", "id": "map", "name": "bpsk"}]})
    layer("profile/t/child", {"base": "profile", "extends": "t/base", "comment": "child",
                              "steps": {"shape": {"pulse": "rrc/0p20"}, "carrier": None,
                                        "+after:timing": [{"sync": "carrier", "id": "loop", "method": "costas", "order": 2}],
                                        "map": {"mapping": "R180"}}})
    got = rp.profile.load("t/child").obj["steps"]
    assert got == [{"wave": "shape", "pulse": "rrc/0p20", "sps": 4}, {"sync": "timing"},
                   {"sync": "carrier", "id": "loop", "method": "costas", "order": 2},
                   {"symbol": "constellation", "id": "map", "name": "bpsk", "mapping": "R180"}]
    assert [s.id for s in rp.profile.load("t/child").steps] == ["shape", "timing", "loop", "map"]

def test_patching_an_id_two_steps_share_is_an_error(layer):
    layer("profile/t/two", {"base": "profile", "comment": "p", "steps": [{"sync": "timing"}, {"sync": "timing"}]})
    layer("profile/t/bad", {"base": "profile", "extends": "t/two", "comment": "c", "steps": {"timing": {"x": 1}}})
    with pytest.raises(ConfigError, match="more than one step 'timing'.*give them"):
        rp.profile.load("t/bad")
    layer("profile/t/none", {"base": "profile", "extends": "t/two", "comment": "c", "steps": {"+after:nope": []}})
    with pytest.raises(ConfigError, match="no step 'nope'"):
        rp.profile.load("t/none")
