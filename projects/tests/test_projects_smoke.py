"""Every program in projects/ compiles, imports (with the sys.path it sets up for itself) and, when it has a command line, answers --help without touching a radio

These are the cheap checks that catch a broken import after a refactor; the decoding logic is covered by the synthetic round trips in test_projects_*.py."""
import pathlib
import subprocess
import sys

import pytest

PROJECTS = pathlib.Path(__file__).resolve().parents[1]
FILES = sorted(p for p in PROJECTS.rglob("*.py") if "__pycache__" not in p.parts and "tests" not in p.relative_to(PROJECTS).parts and not p.name.startswith("_"))
IDS = [str(p.relative_to(PROJECTS)) for p in FILES]
pytest.importorskip("pysdrlib")

def run(code, *args, cwd=None, timeout=120):
    return subprocess.run([sys.executable, "-c", code, *args], capture_output=True, text=True, cwd=cwd, timeout=timeout)

@pytest.mark.parametrize("path", FILES, ids=IDS)
def test_compiles(path):
    compile(path.read_text(encoding="utf-8"), str(path), "exec")

# Import the file the way running it does: its own folder first on sys.path, __name__ not "__main__"
IMPORT = "import runpy, sys; sys.path.insert(0, sys.argv[2]); runpy.run_path(sys.argv[1], run_name='smoke')"

@pytest.mark.parametrize("path", FILES, ids=IDS)
def test_imports(path, tmp_path):
    r = run(IMPORT, str(path), str(path.parent), cwd=tmp_path)
    assert r.returncode == 0, r.stderr[-1500:]

@pytest.mark.parametrize("path", [p for p in FILES if "argparse" in p.read_text(encoding="utf-8") and "__main__" in p.read_text(encoding="utf-8")],
                         ids=lambda p: str(p.relative_to(PROJECTS)))
def test_help(path, tmp_path):
    r = subprocess.run([sys.executable, str(path), "--help"], capture_output=True, text=True, cwd=tmp_path, timeout=120)
    assert r.returncode == 0, r.stderr[-1500:]
    assert "usage" in r.stdout.lower()

