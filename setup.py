"""config/ stays at the root of the repo; the built package carries a copy (rpps/config) so `pip install` has the definitions

The C++ kernels (csrc/rpps_native -> rpps._native) are optional: without pybind11 or a compiler the build warns and
rpps runs its pure Python loops. Build them in place with `python setup.py build_ext -i` (builder.py -c).
"""
import importlib.util
import os
import shutil
import sys
from pathlib import Path

from setuptools import setup
from setuptools.command.build_ext import build_ext
from setuptools.command.build_py import build_py

ROOT = Path(__file__).resolve().parent

class BuildWithConfig(build_py):
    def run(self):
        super().run()
        src = ROOT / "config"
        dst = Path(self.build_lib) / "rpps" / "config"
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__"))

class OptionalBuildExt(build_ext):
    """A failed native build is a warning, not an error: rpps works without rpps._native"""
    def run(self):
        try:
            super().run()
        except Exception as e:
            self._skip(e)

    def build_extension(self, ext):
        try:
            super().build_extension(ext)
        except Exception as e:
            self._skip(e)

    @staticmethod
    def _skip(e):
        if os.environ.get("RPPS_REQUIRE_NATIVE"): # CI and wheel builds: a wheel without the kernels is an error
            raise e
        print(f"WARNING: rpps._native was not built ({e}); rpps will use its Python loops", file=sys.stderr)

def native_extensions():
    try:
        from pybind11.setup_helpers import Pybind11Extension
    except ImportError:
        if os.environ.get("RPPS_REQUIRE_NATIVE"):
            raise
        print("WARNING: pybind11 is not installed; rpps._native is not built", file=sys.stderr)
        return []
    src_dir = ROOT / "csrc" / "rpps_native"
    srcs = sorted(str(p.relative_to(ROOT)) for p in src_dir.glob("*.cpp"))
    args = ["/O2"] if sys.platform == "win32" else ["-O3"]
    macros = [("RPPS_SRC_HASH", source_hash(src_dir))] # rpps._accel warns when the sources have moved on since
    return [Pybind11Extension("rpps._native", srcs, include_dirs=["csrc/rpps_native"], cxx_std=17, extra_compile_args=args,
                              define_macros=macros)]

def source_hash(src_dir):
    """rpps._srchash.source_hash, loaded by path: setup.py cannot import rpps"""
    spec = importlib.util.spec_from_file_location("_rpps_srchash", ROOT / "src" / "rpps" / "_srchash.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.source_hash(src_dir)

setup(
    name="rpps",
    include_package_data=True,
    cmdclass={"build_py": BuildWithConfig, "build_ext": OptionalBuildExt},
    ext_modules=native_extensions(),
)
