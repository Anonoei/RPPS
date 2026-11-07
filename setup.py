"""config/ stays at the root of the repo; the built package carries a copy (rpps/config) so `pip install` has the definitions"""
import shutil
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py

class BuildWithConfig(build_py):
    def run(self):
        super().run()
        src = Path(__file__).resolve().parent / "config"
        dst = Path(self.build_lib) / "rpps" / "config"
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__"))

setup(cmdclass={"build_py": BuildWithConfig})
