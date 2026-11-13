#!/usr/bin/env python
"""Local build helper. Dependencies, extras and tool versions live in pyproject.toml; this only runs the steps.

    python builder.py -s            install the package and the dev extras (editable)
    python builder.py -c            build the C++ kernels in place
    python builder.py -t            run the tests
    python builder.py -d            build the docs into docs/rpps
    python builder.py -b            build the sdist and wheel into dist/
    python builder.py -v p          dry run of a patch version bump (add -r to do it)
    python builder.py -u            upload dist/ to TestPyPI (add -r for PyPI)

Releases are tagged vMAJOR.MINOR.PATCH from a GitHub release, which runs python-publish.yml.
"""
import argparse
import pathlib
import shutil
import subprocess
import sys
import tomllib

ROOT = pathlib.Path(__file__).resolve().parent


def run(*cmd: str) -> None:
    print("Running", " ".join(cmd), flush=True)
    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode:
        sys.exit(result.returncode)


def py(*cmd: str) -> None:
    run(sys.executable, "-m", *cmd)


def project_name() -> str:
    with open(ROOT / "pyproject.toml", "rb") as f:
        return tomllib.load(f)["project"]["name"]


def cmd_setup(args) -> None:
    py("pip", "install", "-e", ".[dev]")


def cmd_compile(args) -> None:
    py("pip", "install", "setuptools>=77", "pybind11>=2.12")  # build_ext runs without build isolation
    run(sys.executable, "setup.py", "build_ext", "-i")


def cmd_local(args) -> None:
    py("pip", "install", "-e", ".")


def cmd_build(args) -> None:
    shutil.rmtree(ROOT / "dist", ignore_errors=True)
    py("build")
    py("twine", "check", *map(str, sorted((ROOT / "dist").glob("*"))))


def cmd_test(args) -> None:
    py("pytest", "-q")


def cmd_docs(args) -> None:
    py("pdoc", "-o", "docs", "--html", "--force", "--template-dir", "pdoc_templates", f"src/{project_name()}")


def cmd_version(args) -> None:
    cmd = ["bumpver", "update", "--allow-dirty", {"M": "--major", "m": "--minor", "p": "--patch"}[args.version]]
    if not args.run:
        cmd += ["--dry", "-n"]
    run(*cmd)


def cmd_upload(args) -> None:
    dist = sorted(map(str, (ROOT / "dist").glob("*")))
    if not dist:
        sys.exit("dist/ is empty: run with -b first")
    cmd = [sys.executable, "-m", "twine", "upload"]
    if not args.run:
        cmd += ["-r", "testpypi"]
    run(*cmd, *dist)


def main() -> None:
    parser = argparse.ArgumentParser(description="Python build helper")
    actions = parser.add_argument_group("Actions", "actions to perform, in the order listed here")
    actions.add_argument("-s", "--setup", action="store_true", help="install the package and the dev extras")
    actions.add_argument("-v", "--version", choices=["M", "m", "p"], help="bump the major, minor or patch version (dry run without -r)")
    actions.add_argument("-c", "--compile", action="store_true", help="build the C++ kernels in place")
    actions.add_argument("-b", "--build", action="store_true", help="build the sdist and wheel")
    actions.add_argument("-l", "--local", action="store_true", help="install the package locally (editable)")
    actions.add_argument("-t", "--test", action="store_true", help="run pytest")
    actions.add_argument("-d", "--docs", action="store_true", help="generate the docs")
    actions.add_argument("-u", "--upload", action="store_true", help="upload dist/ to TestPyPI (PyPI with -r)")
    parser.add_argument("-r", "--run", action="store_true", help="perform the version bump and the upload live")
    args = parser.parse_args()

    if not any((args.setup, args.version, args.compile, args.build, args.local, args.test, args.docs, args.upload)):
        parser.print_help()
        return
    print(f"Running from '{ROOT}'!")

    steps = (
        (args.setup, cmd_setup),
        (args.version, cmd_version),
        (args.compile, cmd_compile),
        (args.build, cmd_build),
        (args.local, cmd_local),
        (args.test, cmd_test),
        (args.docs, cmd_docs),
        (args.upload, cmd_upload),
    )
    for enabled, step in steps:
        if enabled:
            step(args)


if __name__ == "__main__":
    main()
