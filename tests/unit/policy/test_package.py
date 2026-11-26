"""Importing rpps is cheap and side-effect free, and ships what it loads"""
import pytest

import rpps as rp

def test_definitions_ship_with_the_package():
    root = rp._config.PATH_CONFIG
    for kind in ("constellation", "pulse", "filter", "sequence", "scramble", "fec", "interleave", "crc", "frame", "header", "table", "profile"):
        assert any(root.joinpath(kind).rglob("*.json")), kind

def test_extra_config_path_adds_to_the_packaged_definitions(tmp_path, config_layer):
    (tmp_path / "scramble" / "feedthrough").mkdir(parents=True)
    src = rp._config.PATH_CONFIG / "scramble" / "feedthrough" / "v35.json"
    (tmp_path / "scramble" / "feedthrough" / "mine.json").write_text(src.read_text())
    config_layer(tmp_path)
    assert rp.code.scram.load("mine") is not None
    assert rp.code.scram.load("v35") is not None # The packaged ones are still there
    assert {"mine", "v35"} <= set(rp.code.scram.ls()["feedthrough"])

@pytest.mark.parametrize("level", ["INFO", "TRACE"])
def test_logging_is_opt_in(level, capsys):
    log = rp.logger.Logger("t")
    log.info("before")
    assert capsys.readouterr().err == ""
    rp.logger.enable(level)
    try:
        log.info("hello")
        log.trace("fine %s", "detail")
    finally:
        rp.logger.disable()
    err = capsys.readouterr().err
    assert "hello" in err and ("fine detail" in err) == (level == "TRACE")
    log.info("after")
    assert capsys.readouterr().err == ""

def test_a_native_build_from_other_sources_is_stale(tmp_path):
    from types import SimpleNamespace
    from rpps._accel import stale
    from rpps._srchash import source_hash
    (tmp_path / "k.cpp").write_text("int f() { return 1; }\n")
    built = SimpleNamespace(SRC_HASH=source_hash(tmp_path))
    assert stale(built, tmp_path) is None
    (tmp_path / "k.cpp").write_text("int f() { return 2; }\n")
    assert "changed" in stale(built, tmp_path)
    assert "predates" in stale(SimpleNamespace(), tmp_path)
    assert stale(built, tmp_path / "missing") is None # An installed package has no sources to compare
    assert stale(None, tmp_path) is None
