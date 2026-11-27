"""Generate the parts of the documentation that come from the code: API pages and the llms files

    python tools/gen_docs.py             # writes docs/api/*.md, docs/llms.txt, docs/llms-full.txt
    python tools/gen_docs.py --check     # exits 1 if zensical.toml's API nav and the packages disagree

`python tools/builder.py -d` runs this, then `zensical build`. Zensical runs no MkDocs plugins (gen-files), so the pages that
render each package (`::: rpps.<pkg>`, mkdocstrings) are written here, one per top-level package; a nested package gets
its own block on its parent's page, so nothing renders twice.

llms.txt (llmstxt.org) lists the pages; llms-full.txt is AGENTS.md, every public package's docstring and a one-line
signature and summary of every public class and function, so an agent can read the whole library in one file.
"""
import argparse
import ast
import pathlib
import re
import sys
import tomllib

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = ROOT / "src" / "rpps"
DOCS = ROOT / "docs"
SITE = "https://anonoei.github.io/RPPS"

# Top-level packages in the order data runs through them (rpps, Where things are), then the support packages
ORDER = ["source", "signal", "sync", "wave", "symbol", "code", "link", "grid", "profile", "parallel", "sequence", "plan",
         "analysis", "base", "math", "utils", "viz"]
# Private modules worth reading: rendered on the rpps page
ROOT_EXTRA = ["rpps._effort", "rpps.errors"]


def public(name):
    return not name.startswith("_")


def packages(top=SRC):
    """Public packages directly below `top`"""
    return sorted(p for p in top.iterdir() if p.is_dir() and public(p.name) and (p / "__init__.py").exists())


def modname(path):
    return ".".join(path.relative_to(SRC.parent).with_suffix("").parts).removesuffix(".__init__")


def own_members(pkg):
    """What a package's own page block renders: its modules and what its __init__ defines (not nested packages)"""
    mods = [f.stem for f in sorted(pkg.glob("*.py")) if public(f.stem)]
    tree = ast.parse((pkg / "__init__.py").read_text())
    defs = [n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and public(n.name)]
    return mods + defs


def block(pkg):
    members = own_members(pkg)
    lines = [f"::: {modname(pkg / '__init__.py')}", "    options:", "      show_submodules: true"]
    lines.append("      members: " + ("[" + ", ".join(members) + "]" if members else "false"))
    out = ["\n".join(lines)]
    for sub in packages(pkg):
        out += block(sub)
    return out


def first_line(doc):
    """The first paragraph, on one line"""
    para = (doc or "").strip().split("\n\n")[0]
    return " ".join(line.strip() for line in para.split("\n"))


def first_sentence(doc):
    line = first_line(doc)
    m = re.search(r"(?<=[a-z0-9)`])\. ", line)
    return line[:m.start() + 1] if m else line


def docstring(path):
    return ast.get_docstring(ast.parse(path.read_text())) or ""


def api_pages(docs=DOCS):
    tops = {p.name: p for p in packages()}
    missing = set(tops) - set(ORDER)
    if missing:
        sys.exit(f"gen_docs: add {sorted(missing)} to ORDER")
    out = docs / "api"
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*.md"):
        old.unlink()
    rows = ["| Package | What it is |", "|---|---|", f"| [rpps](rpps.md) | {first_line(docstring(SRC / '__init__.py'))} |"]
    root = ["::: rpps\n    options:\n      members: false"] + [f"::: {m}" for m in ROOT_EXTRA]
    (out / "rpps.md").write_text("\n\n".join(root) + "\n")
    for name in ORDER:
        pkg = tops[name]
        (out / f"{name}.md").write_text("\n\n".join(block(pkg)) + "\n")
        rows.append(f"| [rpps.{name}]({name}.md) | {first_line(docstring(pkg / '__init__.py'))} |")
    (out / "index.md").write_text("# API reference\n\nEvery public package, rendered from its docstrings. The design "
                                  "(model, categories, contracts, streaming rules) is [rpps.base](base.md).\n\n"
                                  + "\n".join(rows) + "\n")


def nav_api():
    """The API pages zensical.toml's nav lists"""
    cfg = tomllib.loads((ROOT / "zensical.toml").read_text())
    for entry in cfg["project"]["nav"]:
        if "API" in entry:
            return [p if isinstance(p, str) else next(iter(p.values())) for p in entry["API"]]
    return []


def check_nav():
    want = ["api/index.md", "api/rpps.md"] + [f"api/{n}.md" for n in ORDER]
    have = nav_api()
    if have != want:
        print("zensical.toml nav API should be:\n" + "\n".join(f'    "{p}",' for p in want))
        return False
    return True


# ---- llms files

def signature(node):
    a = node.args
    names = [x.arg for x in a.posonlyargs + a.args if x.arg not in ("self", "cls")]
    if a.vararg:
        names.append("*" + a.vararg.arg)
    elif a.kwonlyargs:
        names.append("*")
    names += [x.arg for x in a.kwonlyargs]
    if a.kwarg:
        names.append("**" + a.kwarg.arg)
    return "(" + ", ".join(names) + ")"


def summaries(path):
    """`name(args): first docstring line` for each public class and function a module defines"""
    tree = ast.parse(path.read_text())
    out = []
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and public(n.name):
            if isinstance(n, ast.ClassDef):
                init = next((b for b in n.body if isinstance(b, ast.FunctionDef) and b.name == "__init__"), None)
                sig = signature(init) if init else ""
            else:
                sig = signature(n)
            out.append(f"- {n.name}{sig}: {first_sentence(ast.get_docstring(n))}".rstrip(": "))
    return out


def walk(pkg):
    """Every public module of a package, the package first"""
    yield pkg / "__init__.py"
    for f in sorted(pkg.glob("*.py")):
        if public(f.stem):
            yield f
    for sub in packages(pkg):
        yield from walk(sub)


def pages():
    cfg = tomllib.loads((ROOT / "zensical.toml").read_text())
    def flat(items, section=""):
        for it in items:
            for title, val in it.items() if isinstance(it, dict) else [(None, it)]:
                if isinstance(val, list):
                    yield from flat(val, title)
                else:
                    yield section, title, val
    return list(flat(cfg["project"]["nav"]))


def h1(path):
    m = re.search(r"^# (.+)$", path.read_text(), re.M)
    return m.group(1) if m else path.stem


def llms(docs=DOCS):
    lines = ["# RPPS", "", "> RF Parameter Processor Suite: a generic signal processor/generator library in Python. "
             "Operators (`*` encode, `/` decode, `+` DSP stage) on data that carries its rates and stream state; whole "
             "signals as JSON profiles; blind analysis of unknown signals.", "",
             f"- [llms-full.txt]({SITE}/llms-full.txt): AGENTS.md, every package docstring and every public signature in one file",
             "- [AGENTS.md](https://github.com/Anonoei/RPPS/blob/main/AGENTS.md): orientation for agents working on the code", ""]
    groups = {}
    for sec, title, page in pages():
        if page.startswith("api/"):
            continue
        url = f"{SITE}/" + re.sub(r"(index)?\.md$", "", page) + ("/" if page not in ("index.md",) and not page.endswith("index.md") else "")
        groups.setdefault(sec or "Docs", []).append(f"- [{title or h1(DOCS / page)}]({url.replace('//', '/').replace(':/', '://')})")
    for sec, items in groups.items():
        lines += ["", f"## {sec}", ""] + items
    lines += ["", "## API", "", f"- [API reference]({SITE}/api/): every package, rendered from its docstrings"]
    for name in ["rpps"] + [f"rpps.{n}" for n in ORDER]:
        path = SRC / "__init__.py" if name == "rpps" else SRC / name.split(".")[1] / "__init__.py"
        page = "rpps" if name == "rpps" else name.split(".")[1]
        lines.append(f"- [{name}]({SITE}/api/{page}/): {first_line(docstring(path))}")
    (docs / "llms.txt").write_text("\n".join(lines) + "\n")

    full = [(ROOT / "AGENTS.md").read_text().rstrip(), "", "# Package docstrings and public API", "",
            "One section per module: the package's docstring in full, then `name(args): summary` per public class and "
            "function. Module docstrings other than packages' are summarized to their first line.", ""]
    mods = [SRC / "__init__.py", SRC / "_effort.py"] + [m for n in ORDER for m in walk(SRC / n)]
    for path in mods:
        name = modname(path)
        doc = docstring(path)
        full.append(f"## {name}")
        full.append("")
        if path.name == "__init__.py" or name == "rpps._effort":
            full.append(re.sub(r"^(#+) ", r"#\1 ", doc.strip(), flags=re.M))   # below this module's heading
        elif doc:
            full.append(first_line(doc))
        items = summaries(path)
        if items:
            full += [""] + items
        full.append("")
    (docs / "llms-full.txt").write_text("\n".join(full).rstrip() + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true", help="only check zensical.toml's API nav")
    a = ap.parse_args(argv)
    if a.check:
        sys.exit(0 if check_nav() else 1)
    api_pages()
    llms()
    if not check_nav():
        sys.exit(1)
    print(f"wrote {DOCS / 'api'}, {DOCS / 'llms.txt'}, {DOCS / 'llms-full.txt'}")


if __name__ == "__main__":
    main()
