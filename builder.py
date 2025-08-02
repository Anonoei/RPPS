import argparse
import importlib
import json
import os
import shutil
import subprocess
import sys
import pathlib

def run(cmd: str, quiet: bool = False):
    print(f"Running '{cmd}'")
    os.system(cmd)

def get_path():
    mod_file = pathlib.Path(os.path.dirname(sys.argv[0])).absolute()
    fpath = subprocess.getoutput(f"cd {str(mod_file)} && git rev-parse --show-toplevel")
    if "fatal:" in fpath:
        fpath = mod_file
    else:
        fpath = pathlib.Path(fpath)
    return fpath

def main():
    parser = argparse.ArgumentParser("Python build helper")

    actions = parser.add_argument_group("Actions", "actions to perform")
    actions.add_argument(
        "-v",
        "--version",
        choices=["M", "m", "p"],
        default=None,
        help="bump version [requires -r]",
    )
    actions.add_argument("-b", "--build", action="store_true", help="build module")
    actions.add_argument("-l", "--local", action="store_true", help="install module locally")
    actions.add_argument("-t", "--test", action="store_true", help="run pytest")
    actions.add_argument("-d", "--docs", action="store_true", help="generate docs")
    actions.add_argument("-u", "--upload", action="store_true", help="upload module [requires -r]")
    actions.add_argument("-c", "--compile", action="store_true")

    parser.add_argument("-r", "--run", action="store_true", help="perform actions live")

    args = parser.parse_args()

    PATH_ROOT = get_path()
    print(f"Running from '{PATH_ROOT}'!")
    os.system(f"cd {PATH_ROOT}")

    mod_names = []

    for p in (PATH_ROOT / "src").iterdir():
        ps = str(p.name)
        if ps.startswith(".") or ps.startswith("_"):
            continue
        elif ps.endswith("egg-info"):
            continue
        mod_names.append(ps)

    args.mods = mod_names

    with open("pyproject.toml", "r") as f:
        dev_deps = None
        deps = None
        for line in f.readlines():
            if line.startswith("dependencies = "):
                deps = line
            if line.startswith("dev = "):
                dev_deps = line
            if deps is not None and dev_deps is not None:
                break
        # deps = '{"deps":' + "" + "}"
        if deps is None or dev_deps is None:
            raise Exception("deps is None")
        deps = json.loads(deps[deps.index("[") - 1 : deps.index("]") + 1])
        dev_deps = json.loads(
            dev_deps[dev_deps.index("[") - 1 : dev_deps.index("]") + 1]
        )
        for dep in deps + dev_deps:
            try:
                importlib.import_module(dep)
            except ModuleNotFoundError:
                os.system(f"{sys.executable} -m pip install {dep}")

    def cmd_local(args):
        run(f"{sys.executable} -m pip install -e .")

    def cmd_build(args):
        dist = PATH_ROOT / "dist"
        if dist.exists():
            shutil.rmtree(dist)
        run(f"{sys.executable} -m build")

    def cmd_compile(args):
        run(f"{sys.executable} setup.py build_ext -i")

    def cmd_upload(args):
        cmd = f"{sys.executable} -m twine upload "
        if not args.run:
            cmd += "-r testpypi dist/*"
        else:
            cmd += "dist/*"
        run(cmd)

    def cmd_test(args):
        run("pytest")

    def cmd_version(args):
        bump = "bumpver update --allow-dirty "
        if not args.run:
            bump += "--dry -n "
        if args.version == "M":
            bump += "--major"
        elif args.version == "m":
            bump += "--minor"
        elif args.version == "p":
            bump += "--patch"
        run(bump)

    def cmd_docs(args):
        run(f"{sys.executable} -m pdoc -o docs --html src/{args.mods[0]} --force")
        docs = PATH_ROOT / "docs"
        if docs.exists():
            mv_docs = docs / args.mods[0]
            if mv_docs.exists():
                temp = PATH_ROOT / ".BUILDpy_TEMP"
                shutil.move(mv_docs, temp)
                shutil.rmtree(docs)
                shutil.move(temp, docs)

    if args.version is not None:
        cmd_version(args)
    if args.compile:
        cmd_compile(args)
    if args.build:
        cmd_build(args)
    if args.local:
        cmd_local(args)
    if args.test:
        cmd_test(args)
    if args.docs:
        cmd_docs(args)
    if args.upload:
        cmd_upload(args)


if __name__ == "__main__":
    main()
