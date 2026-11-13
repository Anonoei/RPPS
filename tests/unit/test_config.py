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
    for f in _config.PATH_CONFIG.rglob("*.json"):
        obj = json.loads(f.read_text())
        assert "base" in obj, f
        domain = f.relative_to(_config.PATH_CONFIG).parts[0]
        assert str(obj["base"]).lower() in _config._DOMAINS[domain].loaders, f

def test_refs_unique_and_aliases_resolve():
    for domain in _config.domains():
        seen = {}
        for cat, ref in refs(domain):
            file = _config.find(domain, ref, cat)
            assert file is not None and file not in seen.values(), (domain, ref)
            seen[(cat, ref)] = file
            for alias in _config.read(file).get("aliases", []):
                assert _config.find(domain, alias, cat) == file, (domain, alias)

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

def test_aliases_resolve():
    assert _config.find("crc", "adsb") == _config.find("crc", "crc24_mode_s") # the name it goes by in a standard
    assert _config.find("fec", "dvbs.3_4_7", "conv") == _config.find("fec", "k7_171_133/3_4", "conv") # system.rate_K
    assert _config.find("fec", "iess308.1_2_7", "conv") == _config.find("fec", "k7_133_171/1_2", "conv")
    assert [s.kind for s in rp.profile.load("dvbs_3_4").steps][-1] == "scramble" # a profile's alias
    assert _config.find("fec", "rs.204_188", "block") is None # a dotted name is an alias or nothing: no family.name guess

def test_family_names_the_loaded_code():
    assert type(rp.code.load("block", "rs/204_188")).__name__ == "rs"
    assert type(rp.code.load("block", "204_188")).__name__ == "rs"

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
        rp.link.load("crc", "mycrc")
        assert "rs/255_223" in rp.code.names("block")
        assert "rs/204_188" in rp.code.names("block")
    finally:
        _config._EXTRA.remove(tmp_path)
