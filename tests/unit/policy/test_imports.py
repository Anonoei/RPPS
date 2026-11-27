"""`import rpps` is cheap, free of side effects and needs no optional package (docs/guide/integrating.md)"""
import subprocess
import sys
import os
from pathlib import Path

import rpps as rp

def test_import_rpps_loads_no_heavy_packages():
    code = ("import sys, rpps; print(' '.join(m for m in ('scipy', 'matplotlib', 'pysdrlib', 'numba', 'torch') "
            "if m in sys.modules))")
    out = subprocess.run([sys.executable, "-W", "ignore", "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == ""

def test_import_has_no_side_effects(tmp_path):
    code = ("import sys, os; import rpps; "
            "bad = [m for m in ('pyboiler', 'matplotlib') if m in sys.modules]; "
            "print(bad, os.path.exists('logs'))")
    env = {**os.environ, "PYTHONPATH": str(Path(rp.__file__).parents[1])}
    out = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, capture_output=True, text=True, env=env)
    assert out.stdout.strip() == "[] False", out.stdout + out.stderr

def test_rpps_imports_without_pysdrlib():
    """pysdrlib is for sources and sinks: importing rpps, and working on codes and definitions, must not need it"""
    import subprocess, sys
    code = ("import builtins; real = builtins.__import__\n"
            "def no(name, *a, **k):\n"
            "    if name.split('.')[0] == 'pysdrlib': raise ImportError(name)\n"
            "    return real(name, *a, **k)\n"
            "builtins.__import__ = no\n"
            "import rpps as rp; rp.load('crc/crc32_mpeg2'); rp.profile.load('ieee802154/2450')")
    subprocess.run([sys.executable, "-c", code], check=True)
