"""The array helpers that replace scipy in the examples (resample_poly, convolve, percentile_filter, least_squares), each checked against
scipy when it is installed (the oracle) and against its definition when it is not"""
import ast
import pathlib

import rpps as rp

def test_rpps_never_imports_scipy():
    """scipy is a test oracle only: nothing under src/ may import it"""
    for path in pathlib.Path(rp.__file__).parent.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""] if isinstance(node, ast.ImportFrom) else []
            assert not any(n.split(".")[0] == "scipy" for n in names), f"{path} imports scipy"

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
