"""`import rpps` stays cheap: numpy and RPPS, not the optional heavy packages (README: What other programs can build on)"""
import subprocess
import sys

def test_import_rpps_loads_no_heavy_packages():
    code = ("import sys, rpps; print(' '.join(m for m in ('scipy', 'matplotlib', 'pysdrlib', 'numba', 'torch') "
            "if m in sys.modules))")
    out = subprocess.run([sys.executable, "-W", "ignore", "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == ""
