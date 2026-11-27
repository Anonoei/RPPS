"""The test suite's own rules (docs/contributing.md, Testing): where a test goes, and what test files share"""
import ast
from pathlib import Path

import rpps as rp
from support.paths import ROOT

UNIT = ROOT/"tests"/"unit"
OWN = {"config", "policy", "standards", "native"} # folders that mirror no package: definitions and CLI, repo-wide rules, whole systems, kernels
FILES = sorted(UNIT.rglob("*.py"))

def folders(root):
    return {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_dir() and any(p.glob("*.py"))}

def test_folders_mirror_the_packages():
    unknown = folders(UNIT) - folders(Path(rp.__file__).parent) - OWN
    assert not unknown, f"tests/unit folders that mirror no src/rpps package: {sorted(unknown)}"

def test_test_files_share_through_support_only():
    """A test file imports no other test file and no conftest: what two need goes in tests/support or a conftest fixture"""
    modules = {p.stem for p in FILES if p.stem.startswith("test_")} | {"conftest"}
    bad = []
    for path in FILES:
        for node in ast.walk(ast.parse(path.read_text())):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""] if isinstance(node, ast.ImportFrom) and not node.level else []
            bad += [f"{path.relative_to(ROOT)}:{node.lineno} imports {n}" for n in names if n.split(".")[0] in modules]
    assert not bad, bad

def test_test_files_are_in_a_folder():
    loose = [p.name for p in UNIT.glob("test_*.py")]
    assert not loose, f"put these in the folder of the package they test: {loose}"
