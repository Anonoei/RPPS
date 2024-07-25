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
    fpath = subprocess.getoutput(
        f"cd {str(mod_file)} && git rev-parse --show-toplevel"
    )
    if "fatal:" in fpath:
        fpath = mod_file
    else:
        fpath = pathlib.Path(fpath)
    return fpath


def main():
    parser = argparse.ArgumentParser("RPPS build helper")

    commands = parser.add_argument_group("Commands")
    commands.add_argument(
        "-l", "--local", action="store_true", help="install RPPS locally"
    )
    commands.add_argument("-b", "--build", action="store_true", help="build RPPS")
    commands.add_argument("-u", "--upload", action="store_true", help="upload RPPS")
    commands.add_argument("-t", "--test", action="store_true", help="run pytest")
    commands.add_argument(
        "-f", "--format", action="store_true", help="run black formatter"
    )
    commands.add_argument("-d", "--docs", action="store_true", help="generate docs")
    commands.add_argument(
        "-p", "--prepare", action="store_true", help="run format, test, and docs"
    )
    commands.add_argument("-v", "--version", action="store_true", help="bump version")

    ver = parser.add_mutually_exclusive_group()

    ver.add_argument("-vM", "--major", action="store_true", help="Bump #.X.X")
    ver.add_argument("-vm", "--minor", action="store_true", help="Bump X.#.X")
    ver.add_argument("-vp", "--patch", action="store_true", help="Bump X.X.#")

    parser.add_argument(
        "-r", "--run", action="store_true", help="actually run the commands and upload"
    )

    args = parser.parse_args()



    PATH_ROOT = get_path()
    os.system(f"cd {PATH_ROOT}")

    with open("pyproject.toml", "r") as f:
        deps = None
        for line in f.readlines():
            if line.startswith("dev = "):
                deps = line
                break
        # deps = '{"deps":' + "" + "}"
        if deps is None:
            raise Exception("deps is None")
        deps = json.loads(deps[deps.index("[") - 1 : deps.index("]") + 1])
        for dep in deps:
            try:
                importlib.import_module(dep)
            except ModuleNotFoundError:
                if dep == "pdoc3":
                    continue
                os.system(f"{sys.executable} -m pip install {dep}")

    def cmd_local(args):
        run(f"{sys.executable} -m pip install -e .")

    def cmd_build(args):
        dist = PATH_ROOT / "dist"
        if dist.exists():
            shutil.rmtree(dist)
        run(f"{sys.executable} -m build")

    def cmd_upload(args):
        cmd = f"{sys.executable} -m twine upload "
        if not args.run:
            cmd += "-r testpypi dist/*"
        else:
            cmd += "dist/*"
        run(cmd)

    def cmd_test(args):
        run("pytest")

    def cmd_format(args):
        run("black src/rpps")

    def cmd_version(args):
        bump = "bumpver update --allow-dirty "
        if not args.run:
            bump += "--dry -n "
        if args.major:
            bump += "--major"
        elif args.minor:
            bump += "--minor"
        elif args.patch:
            bump += "--patch"
        run(bump)

    def cmd_docs(args):
        run(f"{sys.executable} -m pdoc -o docs --html src/rpps --force")
        docs = PATH_ROOT / "docs"
        if docs.exists():
            mv_docs = docs / "rpps"
            if mv_docs.exists():
                temp = PATH_ROOT / ".BUILDpy_TEMP"
                shutil.move(mv_docs, temp)
                shutil.rmtree(docs)
                shutil.move(temp, docs)

    if args.prepare:
        cmd_format(args)
        cmd_test(args)
        cmd_docs(args)

    if args.version:
        cmd_version(args)
    if args.format:
        cmd_format(args)
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
