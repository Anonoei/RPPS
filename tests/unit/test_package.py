"""Importing rpps is cheap and side-effect free, and ships what it loads"""
import os
import subprocess
import sys
from pathlib import Path

import pytest

import rpps as rp

def test_import_has_no_side_effects(tmp_path):
    code = ("import sys, os; import rpps; "
            "bad = [m for m in ('pyboiler', 'matplotlib') if m in sys.modules]; "
            "print(bad, os.path.exists('logs'))")
    env = {**os.environ, "PYTHONPATH": str(Path(rp.__file__).parents[1])}
    out = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, capture_output=True, text=True, env=env)
    assert out.stdout.strip() == "[] False", out.stdout + out.stderr

def test_definitions_ship_with_the_package():
    root = rp._config.PATH_CONFIG
    for kind in ("mod", "coding", "scram", "encap", "chain"):
        assert any(root.joinpath(kind).rglob("*.json")), kind

def test_extra_config_path_adds_to_the_packaged_definitions(tmp_path):
    (tmp_path / "scram" / "fdt").mkdir(parents=True)
    src = rp._config.PATH_CONFIG / "scram" / "fdt" / "v35.json"
    (tmp_path / "scram" / "fdt" / "mine.json").write_text(src.read_text())
    rp.add_config_path(tmp_path)
    try:
        assert rp.scram.load("fdt", "mine") is not None
        assert rp.scram.load("fdt", "v35") is not None # The packaged ones are still there
        assert {"mine", "v35"} <= set(rp.scram.ls()["fdt"])
    finally:
        rp._config._EXTRA.remove(tmp_path)

@pytest.mark.parametrize("level", ["INFO", "TRACE"])
def test_logging_is_opt_in(level, caplog):
    rp.logger.enable(level)
    try:
        rp.logger.Logger("t").info("hello")
    finally:
        rp.logger.disable()
