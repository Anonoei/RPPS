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
            if domain == "chain": # chains are grouped by standard, no category folder
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
            if domain != "chain" and not any(b in ref for b in BIG): # chains are loaded by test_chain
                _config.load(domain, ref, cat, example=True)

def test_bare_names():
    assert _config.find("coding", "204_188", "block") == _config.find("coding", "rs/204_188", "block")
    assert _config.find("scram", "dvb_energy") == _config.find("scram", "adt/dvb_energy")
    with pytest.raises(ConfigError, match="ambiguous"):
        _config.find("coding", "1_2", "conv")

def test_old_names_still_resolve():
    new = _config.find("coding", "rs/204_188", "block")
    assert _config.find("coding", "rs.204_188", "block") == new           # first "." read as "/"
    assert _config.find("coding", "dvbs.3_4_7", "conv") == _config.find("coding", "k7_171_133/3_4", "conv") # declared alias
    assert rp.Chain.load("dvbs_3_4").name == "dvbs_3_4"

def test_family_names_the_loaded_code():
    assert type(rp.coding.load("block", "rs/204_188")).__name__ == "rs"
    assert type(rp.coding.load("block", "204_188")).__name__ == "rs"

def test_user_layer(tmp_path):
    (tmp_path / "encap" / "crc").mkdir(parents=True)
    (tmp_path / "coding" / "block" / "rs").mkdir(parents=True)
    src = _config.find("encap", "crc16_arc")
    (tmp_path / "encap" / "crc" / "mycrc.json").write_text(src.read_text())
    (tmp_path / "encap" / "crc" / "crc16_arc.json").write_text(src.read_text().replace("CRC-16/ARC", "mine"))
    rs = _config.find("coding", "rs/204_188", "block")
    (tmp_path / "coding" / "block" / "rs" / "255_223.json").write_text(rs.read_text().replace("204", "255").replace("188", "223"))
    rp.add_config_path(tmp_path)
    try:
        assert "mycrc" in rp.encap.ls()["crc"]
        assert _config.find("encap", "crc16_arc").parent == tmp_path / "encap" / "crc"
        rp.encap.load("crc", "mycrc")
        assert "rs/255_223" in rp.coding.names("block")
        assert "rs/204_188" in rp.coding.names("block")
    finally:
        _config._EXTRA.remove(tmp_path)
