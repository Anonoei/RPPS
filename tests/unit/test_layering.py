"""Imports run down the chain (README.dev.md, Where things are)

The data-path packages and what they stand on never import the packages built on them: blind analysis (outside the data
path), profiles and the parallel runner (which put operators together). An operator calls rpps.base.control and its
kernel sits beside it, so neither needs them. The one exception is AcquireFEC, the stage that runs the blind code search
as a stream arrives: it imports rpps.analysis.identify inside the call.

Checked on the source (an import of rpps loads every package, so sys.modules cannot tell).
"""
import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2]/"src"/"rpps"
LOWER = ("base", "math", "sequence", "signal", "sync", "wave", "symbol", "code", "link")
UPPER = ("analysis", "profile", "parallel")
LAZY_ALLOWED = {("sync/acquire_fec.py", "analysis")} # (module, package it may import inside a function)

def _imports(path):
    """(package below rpps it names, whether at module level) for each import of rpps in a file"""
    rel = path.relative_to(SRC)
    depth = len(rel.parts) - 1 # Packages between rpps and the file
    tree = ast.parse(path.read_text(encoding="utf-8"))
    top = {id(n) for n in tree.body}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.level:
                up = node.level - 1 # levels above the file's package
                if up < depth: # still inside the same top-level package
                    continue
                if up > depth:
                    continue
                name = (node.module or "").split(".")[0]
                names = [name] if name else [a.name.split(".")[0] for a in node.names]
            elif (node.module or "").startswith("rpps."):
                names = [node.module.split(".")[1]]
            else:
                continue
        elif isinstance(node, ast.Import):
            names = [a.name.split(".")[1] for a in node.names if a.name.startswith("rpps.")]
        else:
            continue
        for n in names:
            yield n, id(node) in top

FILES = sorted(p for pkg in LOWER for p in (SRC/pkg).rglob("*.py"))

@pytest.mark.parametrize("path", FILES, ids=lambda p: p.relative_to(SRC).as_posix())
def test_lower_packages_do_not_import_upward(path):
    rel = path.relative_to(SRC).as_posix() # "/" on every OS, as LAZY_ALLOWED is written
    bad = [(name, top) for name, top in _imports(path) if name in UPPER and not (not top and (rel, name) in LAZY_ALLOWED)]
    assert not bad, f"{rel} imports {', '.join(sorted({n for n, _ in bad}))} (analysis, profile and parallel are built on it)"

def test_checker_sees_relative_imports():
    found = dict(_imports(SRC/"sync"/"acquire_fec.py"))
    assert found.get("analysis") is False # imported, inside a function
