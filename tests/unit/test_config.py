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

def test_every_file_has_header():
    for f in _config.PATH_CONFIG.glob("*/**/*.json"): # Definitions are in a resource folder; config/tags.json is the tag list
        obj = json.loads(f.read_text())
        assert "base" in obj, f
        domain = f.relative_to(_config.PATH_CONFIG).parts[0]
        assert str(obj["base"]).lower() in _config._DOMAINS[domain].loaders, f

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
    assert "apsk/dvbs2_16apsk_3_4" in dvbs2["constellation"] # inherited from its parent
    assert _config.find("fec", "rs.204_188", "block") is None # no family.name guess

def test_a_loaded_code_is_named_by_its_ref():
    assert type(rp.code.load("block/rs/204_188")).__name__ == "rs/204_188"
    assert type(rp.code.load("block/204_188")).__name__ == "rs/204_188"
    assert rp.code.load("conv/k7_171_133/3_4").name != rp.code.load("conv/k7_171_133/1_2").name

def test_user_layer(tmp_path):
    (tmp_path / "crc").mkdir(parents=True)
    (tmp_path / "fec" / "block" / "rs").mkdir(parents=True)
    src = _config.find("crc", "crc16_arc")
    (tmp_path / "crc" / "mycrc.json").write_text(src.read_text())
    (tmp_path / "crc" / "crc16_arc.json").write_text(src.read_text().replace("CRC-16/ARC", "mine"))
    rs = _config.find("fec", "rs/204_188", "block")
    (tmp_path / "fec" / "block" / "rs" / "255_223.json").write_text(rs.read_text().replace("204", "255").replace("188", "223"))
    rp.add_config_path(tmp_path)
    try:
        assert "mycrc" in rp.link.ls()["crc"]
        assert _config.find("crc", "crc16_arc").parent == tmp_path / "crc"
        rp.link.load("crc/mycrc")
        assert "rs/255_223" in rp.code.names("block")
        assert "rs/204_188" in rp.code.names("block")
    finally:
        _config._EXTRA.remove(tmp_path)

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
