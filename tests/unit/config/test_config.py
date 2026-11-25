import json
import pytest
import rpps as rp
from rpps import _config
from rpps.errors import ConfigError

BIG = ("dvbs2_64800",) # Large definitions: only checked for a header, not built

def refs(domain):
    """(category, reference) of every definition of a domain"""
    for key, names in _config.ls(domain).items():
        cat, _, group = key.partition("/")
        for n in names:
            if domain == "profile": # profiles are grouped by family, no category folder
                yield None, f"{key}/{n}"
            else:
                yield cat, f"{group}/{n}" if group else n

def test_the_folder_says_the_base():
    for f in _config.PATH_CONFIG.glob("*/**/*.json"): # Definitions are in a resource folder; config/tags.json is the tag list
        assert "base" not in json.loads(f.read_text()), f
        domain = f.relative_to(_config.PATH_CONFIG).parts[0]
        assert _config.file_base(f) in _config._DOMAINS[domain].loaders, f
    assert _config.file_base(_config.find("fec", "rs/204_188", "block")) == "block"
    assert _config.file_base(_config.find("plan", "dvbrcs2/forward", "builder")) == "builder"
    assert _config.file_base(_config.find("plan", "demo/mftdma")) == "plan"
    assert _config.file_base(_config.find("profile", "dvbs2/normal")) == "profile"
    assert _config.base_of("filter", ("iir", "sos", "x")) == "iir"
    with pytest.raises(ConfigError, match="one of the folders"):
        _config.base_of("fec", ("ldpc", "x")) # fec has no loader of its own name: block/ or conv/

def test_a_file_that_says_its_base_is_refused(tmp_path, config_layer):
    d = tmp_path / "crc"
    d.mkdir()
    (d / "said.json").write_text(json.dumps({"base": "crc", "type": "crc", "comment": "x", "width": 8, "poly": "0x07",
                                             "init": "0x00", "refin": False, "refout": False, "xorout": "0x00"}))
    config_layer(tmp_path)
    with pytest.raises(ConfigError, match="folder the file is in"):
        rp.link.load("crc/said")

def test_save_leaves_base_to_the_folder(tmp_path, config_layer):
    obj = {"base": "constellation", "comment": "two points", "points": [[1, 0], [-1, 0]]}
    f = _config.save("constellation", "constellation", "mine/two", obj, dest=tmp_path)
    assert f == tmp_path / "constellation" / "mine" / "two.json" and "base" not in json.loads(f.read_text())
    with pytest.raises(ConfigError, match="not in a config folder"):
        _config.expand(f) # (outside every layer, nothing says it is a constellation)
    config_layer(tmp_path)
    assert _config.expand(f)["base"] == "constellation"
    assert len(rp.symbol.load("mine/two").points) == 2
    f = _config.save("filter", "fir", "taps/x0.5", {"base": "fir", "type": "taps", "taps": [1]}, dest=tmp_path)
    assert f.name == "x0.5.json"
    with pytest.raises(ConfigError, match="folder says 'fir'"):
        _config.save("filter", "fir", "taps/y", {"base": "iir", "type": "taps"}, dest=tmp_path)

def test_refs_unique():
    for domain in _config.domains():
        seen = {}
        for cat, ref in refs(domain):
            file = _config.find(domain, ref, cat)
            assert file is not None and file not in seen.values(), (domain, ref)
            seen[(cat, ref)] = file

def test_no_aliases():
    """Other names for a definition are gone: its tags say which standards use it"""
    for f in _config.PATH_CONFIG.glob("*/**/*.json"):
        assert "aliases" not in json.loads(f.read_text()), f

def test_every_file_loads():
    for domain in _config.domains():
        for cat, ref in refs(domain):
            if not any(b in ref for b in BIG):
                _config.load(domain, ref, cat, example=True)

def test_bare_names():
    assert _config.find("fec", "204_188", "block") == _config.find("fec", "rs/204_188", "block")
    assert _config.find("scramble", "dvb_energy") == _config.find("scramble", "additive/dvb_energy")
    with pytest.raises(ConfigError, match="ambiguous"):
        _config.find("fec", "1_2", "conv")

def test_tags_say_which_standard():
    assert "crc24_mode_s" in _config.tagged("adsb")["crc"]
    assert "conv/k7_171_133/3_4" in _config.tagged("dvb-s")["fec"] and "conv/k7_133_171/1_2" in _config.tagged("iess-308")["fec"]
    dvbs2 = _config.tagged("dvb-s2")
    assert {"constellation", "fec", "interleave", "header", "table", "profile"} <= set(dvbs2)
    assert "apsk/dvbs2_16apsk" in dvbs2["constellation"]
    assert _config.find("fec", "rs.204_188", "block") is None # no family.name guess

def test_a_loaded_code_is_named_by_its_ref():
    assert type(rp.code.load("block/rs/204_188")).__name__ == "rs/204_188"
    assert type(rp.code.load("block/204_188")).__name__ == "rs/204_188"
    assert rp.code.load("conv/k7_171_133/3_4").name != rp.code.load("conv/k7_171_133/1_2").name

def test_user_layer(tmp_path, config_layer):
    (tmp_path / "crc").mkdir(parents=True)
    (tmp_path / "fec" / "block" / "rs").mkdir(parents=True)
    src = _config.find("crc", "crc16_arc")
    (tmp_path / "crc" / "mycrc.json").write_text(src.read_text())
    (tmp_path / "crc" / "crc16_arc.json").write_text(src.read_text().replace("CRC-16/ARC", "mine"))
    rs = _config.find("fec", "rs/204_188", "block")
    (tmp_path / "fec" / "block" / "rs" / "255_223.json").write_text(rs.read_text().replace("204", "255").replace("188", "223"))
    config_layer(tmp_path)
    assert "mycrc" in rp.link.ls()["crc"]
    assert _config.find("crc", "crc16_arc").parent == tmp_path / "crc"
    rp.link.load("crc/mycrc")
    assert "rs/255_223" in rp.code.names("block")
    assert "rs/204_188" in rp.code.names("block")

# ---- Choices and "when"
VARIANT = {"params": {"modulation": {"default": "qpsk", "choices": {"qpsk": {"shape": "QPSK", "bits": 2},
                                                                    "16qam": {"shape": "16QAM", "bits": 4}}},
                      "rolloff": {"default": 0.35, "choices": {"0.2": {"pulse": "rrc/0p20"}, "0.35": {"pulse": "rrc/0p35"}}},
                      "m": {"default": 2, "choices": [2, 4]}},
           "steps": [{"pulse": {"param": "rolloff.pulse"}},
                     {"agc": True, "when": "modulation == '16qam'"},
                     {"shape": {"param": "modulation.shape"}, "n": {"expr": "modulation.bits*m"}}],
           "bw": {"expr": "(1 + rolloff)*10"}}

def test_choice_bundles_fill_their_fields():
    out = _config.resolve(VARIANT)
    assert out["steps"] == [{"pulse": "rrc/0p35"}, {"shape": "QPSK", "n": 4}] and out["bw"] == pytest.approx(13.5)
    out = _config.resolve(VARIANT, {"modulation": "16qam", "rolloff": 0.2, "m": 4})
    assert out["steps"] == [{"pulse": "rrc/0p20"}, {"agc": True}, {"shape": "16QAM", "n": 16}]
    assert _config.declared(out) == {"modulation": "16qam", "rolloff": 0.2, "m": 4}

def test_a_value_not_a_choice_is_an_error():
    for bad in ({"modulation": "8psk"}, {"rolloff": 0.5}, {"m": 3}):
        with pytest.raises(ConfigError, match="not one of its choices"):
            _config.resolve(VARIANT, bad)
    with pytest.raises(ConfigError, match="not a choice's field"):
        _config.resolve({**VARIANT, "x": {"expr": "modulation.nope"}})
    with pytest.raises(ConfigError, match="not a parameter or a choice's field"):
        _config.resolve({**VARIANT, "x": {"param": "modulation.nope"}})

def test_when_without_params():
    obj = {"steps": [{"a": 1, "when": "1 > 2"}, {"b": 2, "when": "'x' == 'x'"}]}
    assert _config.resolve(obj)["steps"] == [{"b": 2}]

def test_a_child_narrows_the_choices():
    child = _config.merge(VARIANT, {"params": {"modulation": {"choices": {"qpsk": {"shape": "QPSK", "bits": 2}}}}})
    with pytest.raises(ConfigError, match="not one of its choices"):
        _config.resolve(child, {"modulation": "16qam"})

def test_param_types_and_ranges():
    obj = {"params": {"sps": {"default": 4, "type": "int", "min": 2, "max": 64}, "beta": {"default": 0.35, "type": "float"},
                      "name": {"default": "x", "type": "str"}}, "v": {"param": "sps"}}
    assert _config.resolve(obj, {"beta": 1})["v"] == 4 # an int is a float
    for bad, msg in (({"sps": "fast"}, "not an int"), ({"sps": 4.0}, "not an int"), ({"sps": True}, "not an int"),
                     ({"sps": 1}, "below its minimum"), ({"sps": 65}, "above its maximum"), ({"name": 3}, "not a str")):
        with pytest.raises(ConfigError, match=msg):
            _config.resolve(obj, bad)
    with pytest.raises(ConfigError, match="type is one of"):
        _config.resolve({"params": {"a": {"default": 1, "type": "integer"}}})

def test_expression_functions():
    v = {"n": 10, "k": 3}
    assert _config.evaluate("ceil(n/k)", v) == 4 and _config.evaluate("floor(n/k)", v) == 3
    assert _config.evaluate("round(n/4)", v) == 2 and _config.evaluate("log2(8)", v) == 3
    with pytest.raises(ConfigError):
        _config.evaluate("open('x')", v)

def test_every_variant_of_every_profile_loads():
    """Each combination of a shipped profile's choices loads and builds its receiver"""
    for fam, modes in rp.profile.ls().items():
        for mode in modes:
            name = f"{fam}/{mode}"
            for params in rp.profile.variants(name):
                assert rp.profile.load(name, **params).rx() is not None, (name, params)

def test_format_is_stable_and_fits():
    for f in _config.PATH_CONFIG.rglob("*.json"):
        obj = json.loads(f.read_text())
        text = _config.format_json(obj) + "\n"
        assert text == f.read_text(), f"{f}: run python -m rpps fmt"
        assert _config.format_json(json.loads(text)) + "\n" == text, f
    obj = {"base": "x", "steps": [{"sync": "timing"}, {"wave": "shape", "sps": {"param": "sps"}}],
           "rows": [[1, 2], [3, 4]], "taps": list(range(100))}
    text = _config.format_json(obj)
    assert '  "steps": [{"sync": "timing"}, {"wave": "shape", "sps": {"param": "sps"}}],' in text
    assert '    [1, 2],\n    [3, 4]\n' in text # rows: one a line
    assert f'  "taps": {json.dumps(list(range(100)))}' in text # numbers: one line, however long

def test_comment_lines():
    long = "One clause about a thing; " * 12 + "the end"
    text = _config.format_json({"base": "x", "comment": long, "params": {"a": {"default": 1, "comment": long.upper()}}})
    obj = json.loads(text)
    assert isinstance(obj["comment"], list) and all(len(json.dumps(s)) <= _config.WIDTH for s in obj["comment"])
    assert obj["comment"][0].endswith(";") # cut where a clause ends
    assert _config.join_comments(obj) == {"base": "x", "comment": long,
                                          "params": {"a": {"default": 1, "comment": long.upper()}}}
    assert json.loads(_config.format_json({"comment": ["short", "one"]}))["comment"] == "short one" # fits: a string
    with pytest.raises(ConfigError, match="list of strings"):
        _config.join_comments({"comment": ["a", 1]})

def test_a_comment_list_loads_as_one_string(tmp_path, config_layer):
    d = tmp_path / "sequence" / "word"
    d.mkdir(parents=True)
    (d / "w.json").write_text(json.dumps({"type": "word", "comment": ["The word,", "MSB first"],
                                          "word": "0xF3A2", "bits": 16, "order": "msb"}))
    config_layer(tmp_path)
    assert _config.expand(d / "w.json")["comment"] == "The word, MSB first"
