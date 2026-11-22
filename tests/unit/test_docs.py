"""The documentation: its code runs, every example is shown, every package documents itself, the llms files build

Python blocks in README.md and docs/ run here, a file's blocks in one namespace in order, unless a block holds
`# docs: skip` (it needs a capture or a radio). A block that only embeds an example (`--8<--`) is the example's, run by
test_examples.py.
"""
import ast
import importlib.util
import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"
GENERATED = {DOCS / "api", DOCS / "llms.txt", DOCS / "llms-full.txt"}

needs_source = pytest.mark.skipif(not (ROOT / "src" / "rpps").is_dir() or not (ROOT / "scripts").is_dir(),
                                  reason="the source tree (a wheel test has only tests, examples and docs)")


def pages():
    files = [ROOT / "README.md"]
    if DOCS.is_dir():
        files += sorted(p for p in DOCS.rglob("*.md") if not any(g == p or g in p.parents for g in GENERATED))
    return files


def runnable_blocks(path):
    blocks = re.findall(r"^```python\n(.*?)^```", path.read_text(), re.S | re.M)
    return [b for b in blocks if "# docs: skip" not in b and not b.lstrip().startswith("--8<--")]


@pytest.mark.parametrize("path", [p for p in pages() if runnable_blocks(p)], ids=lambda p: str(p.relative_to(ROOT)))
def test_python_blocks_run(path, tmp_path, monkeypatch):
    blocks = runnable_blocks(path)
    monkeypatch.chdir(tmp_path) # anything a block writes lands here
    ns = {"__name__": "docs"}
    for code in blocks:
        exec(compile(code, str(path), "exec"), ns)


def test_every_example_is_shown():
    """Each script in examples/ is embedded in a guide or linked from the examples page"""
    text = "\n".join(p.read_text() for p in pages())
    for script in sorted((ROOT / "examples").glob("*.py")):
        shown = f'--8<-- "examples/{script.name}"' in text or f"examples/{script.name}" in text or \
            f"| {script.name} |" in text
        assert shown, f"examples/{script.name} is in no guide and not on docs/examples.md"


def test_snippets_exist():
    for page in pages():
        for ref in re.findall(r'^--8<-- "([^"]+)"', page.read_text(), re.M):
            assert (ROOT / ref).is_file(), f"{page.name} embeds {ref}, which is gone"


@needs_source
def test_every_package_documents_itself():
    """A public package's docstring is its documentation: no package without one, none pointing at removed files"""
    for init in sorted((ROOT / "src" / "rpps").rglob("__init__.py")):
        rel = init.relative_to(ROOT / "src")
        if any(part.startswith("_") for part in rel.parts[:-1]):
            continue
        doc = ast.get_docstring(ast.parse(init.read_text()))
        assert doc, f"{rel} has no docstring"
    for py in (ROOT / "src" / "rpps").rglob("*.py"):
        assert "README.dev.md" not in py.read_text(), f"{py.relative_to(ROOT)} cites README.dev.md: name a docstring"


@needs_source
def test_gen_docs(tmp_path):
    """The API nav in zensical.toml matches the packages, and the llms files build and stay readable in one go"""
    spec = importlib.util.spec_from_file_location("gen_docs", ROOT / "scripts" / "gen_docs.py")
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)
    assert gen.check_nav()
    gen.api_pages(tmp_path)
    assert {p.stem for p in (tmp_path / "api").glob("*.md")} == {"index", "rpps", *gen.ORDER}
    gen.llms(tmp_path)
    full = (tmp_path / "llms-full.txt").read_text()
    assert full.startswith((ROOT / "AGENTS.md").read_text().rstrip()[:200])
    assert "## rpps.base" in full and "## rpps.sync.track" in full
    assert len(full) < 240_000, f"llms-full.txt is {len(full)} bytes: about {len(full)//4} tokens"
