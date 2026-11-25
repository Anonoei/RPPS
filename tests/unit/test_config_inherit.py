"""Definitions that extend others (parents and children), abstract parents, and tags"""
import json

import numpy as np
import pytest

import rpps as rp
from rpps import _config
from rpps.errors import ConfigError

@pytest.fixture
def layer(tmp_path, config_layer):
    """A user config layer: write(path, obj) puts a definition in it"""
    def write(rel, obj):
        f = tmp_path / f"{rel}.json"
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(obj))
        return f
    config_layer(tmp_path)
    return write

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
    layer("constellation/t/base", {"comment": "QPSK", "tags": ["dvb-s2"], "rings": [{"n": 4}],
                                   "maps": [{"map": [0, 1, 3, 2], "comment": "R000"}]})
    layer("constellation/t/axes", {"extends": "t/base", "comment": "on the axes", "rotate": -45})
    a, b = rp.symbol.load("t/base"), rp.symbol.load("t/axes")
    assert np.allclose(b.points, a.points*np.exp(-1j*np.pi/4))
    assert np.array_equal(a.mapping.arr, b.mapping.arr)
    assert _config.file_tags(_config.find("constellation", "t/axes")) == ["dvb-s2"]

def test_extends_is_looked_up_below_the_category_first(layer):
    layer("fec/block/rs/fam", {"type": "rs", "abstract": True, "comment": "family", "m": 8, "poly": 285})
    layer("fec/block/rs/fam_15_11", {"extends": "rs/fam", "comment": "a member", "n": 15, "k": 11})
    c = rp.code.load("block/rs/fam_15_11")
    assert c.encoder.num == 11 and c.encoder.den == 15

def test_abstract_parents_are_hidden_and_refuse_to_load(layer):
    layer("crc/abs_parent", {"abstract": True, "comment": "half a CRC", "width": 16, "poly": "0x1021"})
    layer("crc/abs_child", {"extends": "abs_parent", "type": "crc", "comment": "XMODEM", "init": "0x0",
                            "xorout": "0x0", "refin": False, "refout": False, "check": "0x31c3"})
    assert "abs_parent" not in _config.ls("crc")["crc"] and "abs_parent" in _config.ls("crc", all=True)["crc"]
    with pytest.raises(ConfigError, match="abstract"):
        rp.link.load("crc/abs_parent")
    assert rp.link.load("crc/abs_child").encoder.compute(b"123456789") == 0x31c3

def test_loops_and_other_resources_are_refused(layer):
    layer("crc/loop_a", {"extends": "loop_b", "comment": "a"})
    layer("crc/loop_b", {"extends": "loop_a", "comment": "b"})
    with pytest.raises(ConfigError, match="loops"):
        _config.read(_config.find("crc", "loop_a"))
    layer("crc/cross", {"extends": "constellation/psk/qpsk", "comment": "wrong kind"})
    with pytest.raises(ConfigError, match="its own kind"):
        _config.read(_config.find("crc", "cross"))
    layer("crc/orphan", {"extends": "no_such_parent", "comment": "?"})
    with pytest.raises(ConfigError, match="not a crc definition"):
        _config.read(_config.find("crc", "orphan"))

def test_the_raw_file_keeps_extends(layer):
    f = layer("crc/raw_child", {"extends": "crc16_xmodem", "comment": "same"})
    assert _config.read_raw(f) == {"extends": "crc16_xmodem", "comment": "same"}
    assert _config.read(f)["poly"] == _config.read(_config.find("crc", "crc16_xmodem"))["poly"]

def test_every_tag_used_is_declared():
    known = set(_config.tags())
    used = set(_config.tagged(all=True))
    assert used <= known, f"undeclared tags (add them to config/tags.json): {sorted(used - known)}"

def test_tagged_lists_across_resources(layer):
    layer("crc/tagged_x", {"extends": "crc16_xmodem", "comment": "x", "tags": ["test-tag"]})
    found = _config.tagged("test-tag")
    assert found == {"crc": ["tagged_x"]}
    with pytest.raises(ConfigError, match="No tag"):
        _config.tagged("no-such-tag-anywhere")

def test_save_needs_a_destination(monkeypatch):
    monkeypatch.setattr(_config, "_EXTRA", [])
    with pytest.raises(ConfigError, match="destination"):
        _config.save("crc", "crc", "x", {"base": "crc"})

def test_a_child_patches_single_steps(layer):
    layer("profile/t/base", {"comment": "parent", "params": {"sps": {"default": 4}},
                             "steps": [{"wave": "shape", "pulse": "rrc/0p35", "sps": {"param": "sps"}}, {"sync": "timing"},
                                       {"sync": "carrier", "order": 2, "constellation": "bpsk"},
                                       {"symbol": "constellation", "id": "map", "name": "bpsk"}]})
    layer("profile/t/child", {"extends": "t/base", "comment": "child",
                              "steps": {"shape": {"pulse": "rrc/0p20"}, "carrier": None,
                                        "+after:timing": [{"sync": "carrier", "id": "loop", "method": "mpower", "order": 2}],
                                        "map": {"mapping": "R180"}}})
    got = rp.profile.load("t/child").obj["steps"]
    assert got == [{"wave": "shape", "pulse": "rrc/0p20", "sps": 4}, {"sync": "timing"},
                   {"sync": "carrier", "id": "loop", "method": "mpower", "order": 2},
                   {"symbol": "constellation", "id": "map", "name": "bpsk", "mapping": "R180"}]
    assert [s.id for s in rp.profile.load("t/child").steps] == ["shape", "timing", "loop", "map"]

def test_patching_an_id_two_steps_share_is_an_error(layer):
    layer("profile/t/two", {"comment": "p", "steps": [{"sync": "timing"}, {"sync": "timing"}]})
    layer("profile/t/bad", {"extends": "t/two", "comment": "c", "steps": {"timing": {"x": 1}}})
    with pytest.raises(ConfigError, match="more than one step 'timing'.*give them"):
        rp.profile.load("t/bad")
    layer("profile/t/none", {"extends": "t/two", "comment": "c", "steps": {"+after:nope": []}})
    with pytest.raises(ConfigError, match="no step 'nope'"):
        rp.profile.load("t/none")

FAM = {"abstract": True, "type": "rs", "comment": "family", "tags": ["dvb-s"], "m": 8, "poly": 285,
       "variants": {"columns": ["name", "n", "k", "comment", "tags"],
                    "rows": [["v_15_11", 15, 11, "short one", ["ccsds"]], ["v_204_188", 204, 188, "DVB-S", []]]}}

def test_variants_load_as_children_of_their_parent(layer):
    layer("fec/block/rs/vfam", FAM)
    assert "v_15_11" in _config.ls("fec")["block/rs"] and "vfam" not in _config.ls("fec")["block/rs"]
    code = rp.code.load("block/rs/v_204_188")
    assert code.name == "rs/v_204_188" and (code.e_num, code.e_den) == (188, 204)
    assert rp.code.load("v_15_11").e_den == 15 # (a bare name, as a file's)
    obj = _config.read(_config.find("fec", "rs/v_15_11", "block"))
    assert obj["base"] == "block" and obj["comment"] == "short one" and obj["m"] == 8 and "variants" not in obj
    assert _config.read_raw(_config.find("fec", "rs/v_15_11", "block")) == {"extends": "vfam", "n": 15, "k": 11,
                                                                              "comment": "short one", "tags": ["ccsds"]}
    assert "block/rs/v_15_11" in rp.tagged("ccsds")["fec"] and "block/rs/v_204_188" in rp.tagged("dvb-s")["fec"]
    with pytest.raises(ConfigError, match="abstract"):
        rp.code.load("block/rs/vfam")

def test_a_file_extends_a_variant_and_a_variant_follows_its_parent(layer):
    f = layer("fec/block/rs/vfam2", {**FAM, "variants": {"columns": ["name", "n", "k"], "rows": [["w_15_11", 15, 11]]}})
    layer("fec/block/rs/w_child", {"extends": "rs/w_15_11", "comment": "of a variant", "k": 9})
    assert (rp.code.load("block/rs/w_child").e_num, rp.code.load("block/rs/w_child").e_den) == (9, 15)
    import os
    f.write_text(json.dumps({**FAM, "variants": {"columns": ["name", "n", "k"], "rows": [["w_15_11", 15, 7]]}}))
    os.utime(f, ns=(f.stat().st_atime_ns, f.stat().st_mtime_ns + 10**9)) # (a later change, however coarse the clock)
    assert rp.code.load("block/rs/w_15_11").e_num == 7

def test_a_dotted_column_sets_a_key_inside_and_a_loadable_parent_keeps_its_variants(layer):
    layer("constellation/t/vring", {"comment": "a ring", "rings": [{"n": 4}], "params": {"turn": {"default": 0}},
                                    "rotate": {"param": "turn"},
                                    "variants": {"columns": ["name", "params.turn", "comment"],
                                                 "rows": [["vring_45", 45, "turned"]]}})
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error") # ("variants" is not a key the loader is left to read)
        a, b = rp.symbol.load("t/vring"), rp.symbol.load("t/vring_45")
    assert np.allclose(b.points, a.points*np.exp(1j*np.pi/4))

@pytest.mark.parametrize("variants, match", [
    ({"columns": ["n", "name"], "rows": []}, "takes"),
    ({"columns": ["name", "n"], "rows": [["x1", 1, 2]]}, "does not have"),
    ({"columns": ["name", "n"], "rows": [["x1", 1], ["X1", 2]]}, "twice"),
    ({"columns": ["name", "extends"], "rows": [["x1", "a"]]}, "not a column"),
    ({"columns": ["name", "n"], "rows": [["a b", 1]]}, "not a name"),
])
def test_bad_variants_are_refused(layer, variants, match):
    layer("crc/vbad", {"abstract": True, "comment": "x", "variants": variants})
    with pytest.raises(ConfigError, match=match):
        _config.ls("crc")

def test_a_variant_that_is_also_a_file_is_refused(layer):
    layer("crc/vdup_parent", {"abstract": True, "comment": "x", "variants": {"columns": ["name"], "rows": [["vdup"]]}})
    layer("crc/vdup", {"extends": "crc16_xmodem", "comment": "a file"})
    with pytest.raises(ConfigError, match="also a file"):
        _config.ls("crc")

def test_examples_load_and_list_but_are_never_blind_candidates(layer):
    from rpps.profile.catalogue import candidates
    layer("crc/ex_crc", {"extends": "crc16_xmodem", "comment": "made up", "tags": ["example"]})
    layer("crc/ex_child", {"extends": "ex_crc", "comment": "a child of one is one too"})
    layer("profile/ex/beacon", {"extends": "ax25/afsk1200", "comment": "made up", "tags": ["example"]})
    assert {"ex_crc", "ex_child"} <= set(rp.link.ls()["crc"]) and "beacon" in rp.profile.ls()["ex"]
    assert not {"ex_crc", "ex_child"} & set(rp.link.ls(blind=True)["crc"])
    assert "ex" not in rp.profile.ls(blind=True) and not [n for n, _ in candidates() if n.startswith("ex/")]
    assert rp.link.load("crc/ex_child") is not None and rp.profile.load("ex/beacon") is not None
    for family, modes in rp.profile.ls(blind=True).items(): # (the shipped demos too)
        assert family != "demo", modes

def test_a_higher_layer_decides_what_a_name_lists_as(layer):
    layer("crc/crc16_arc", {"extends": "crc16_xmodem", "comment": "mine, made up", "tags": ["example"]})
    assert "crc16_arc" in rp.link.ls()["crc"] and "crc16_arc" not in rp.link.ls(blind=True)["crc"]

def test_only_the_users_definitions(layer, monkeypatch):
    """use_shipped(False): only the user's directories are searched; a shipped name is not found"""
    layer("crc/mine_only", {"comment": "mine", "width": 8, "poly": 7, "init": 0, "refin": False, "refout": False, "xorout": 0})
    layer("profile/mine/beacon", {"comment": "mine", "steps": [{"symbol": "constellation", "name": "mine_bpsk"}]})
    layer("constellation/psk/mine_bpsk", {"comment": "mine", "rings": [{"n": 2, "pha": 0}]})
    monkeypatch.setattr(_config, "_SHIPPED", False)
    assert rp.link.ls()["crc"] == ["mine_only"] and rp.profile.ls() == {"mine": ["beacon"]}
    assert rp.symbol.ls() == {"psk": ["mine_bpsk"]} and rp.link.load("crc/mine_only") is not None
    with pytest.raises(Exception):
        rp.symbol.load("psk/qpsk")
    assert "dvb-s2" in _config.tags() # The shipped tags still name tags
    assert "crc16_xmodem" not in str(_config.tagged())

def test_selected_tags_narrow_what_the_blind_tools_try(monkeypatch):
    from rpps.profile.catalogue import candidates
    monkeypatch.setattr(_config, "_SELECTED", None)
    every = {n for n, _ in candidates()}
    rp.select_tags(["dvb-s2"])
    try:
        tagged = set(_config.tagged("dvb-s2")["profile"])
        assert {f"{f}/{m}" for f, ms in rp.profile.ls(blind=True).items() for m in ms} == tagged and len(rp.profile.ls()) > 5
        assert {n for n, _ in candidates()} == every & tagged
        untagged = [n for n in rp.link.ls()["crc"] if not _config.file_tags(_config.find("crc", n))]
        assert untagged and set(untagged) <= set(rp.link.ls(blind=True)["crc"]) # the untagged generic ones stay
        assert rp.profile.load("lte/downlink") is not None # loads by name whatever is selected
        with pytest.raises(ConfigError, match="no tag"):
            rp.select_tags(["not-a-tag"])
    finally:
        rp.select_tags(None)
    assert {n for n, _ in candidates()} == every
