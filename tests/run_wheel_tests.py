"""Run the fast unit tests against the *installed* rpps (used by cibuildwheel's test-command).

The tests are copied away from the checkout so `import rpps` can only find the installed wheel; the docs tests read
README.md and pyproject.toml from the working directory."""
import pathlib
import shutil
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent

with tempfile.TemporaryDirectory() as tmp:
    run = pathlib.Path(tmp)
    shutil.copytree(ROOT / "tests", run / "tests", ignore=shutil.ignore_patterns("__pycache__", "dev", "prop"))
    for name in ("README.md", "pyproject.toml"):
        shutil.copy(ROOT / name, run / name)
    sys.exit(subprocess.call([sys.executable, "-m", "pytest", "-q", "-m", "not slow", "tests/unit"], cwd=run))
