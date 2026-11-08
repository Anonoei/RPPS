import os
import sys
import json
import logging
from pathlib import Path

from .errors import ConfigError

BYTE_ORDER = ">" if sys.byteorder == "big" else "<"

# Definitions (mod/coding/scram/encap/chain) ship with the package. A directory in $RPPS_CONFIG, or
# one added with `add_config_path`, is searched first, so a project can add or override definitions.
# In a checkout that's config/ next to src/; an installed package carries a copy next to its modules (setup.py puts it there)
_HERE = Path(__file__).resolve().parent
PATH_CONFIG = _HERE.parents[1] / "config" if (_HERE.parents[1] / "config").is_dir() else _HERE / "config"
_EXTRA: list = [Path(p) for p in os.environ.get("RPPS_CONFIG", "").split(os.pathsep) if p]

def add_config_path(path):
    """Search `path` (laid out like the shipped config/) before the packaged definitions"""
    _EXTRA.insert(0, Path(path))

def layers(kind):
    """Directories holding definitions of a kind ("mod", "coding", ...), highest priority first"""
    return [d / kind for d in (*_EXTRA, PATH_CONFIG) if (d / kind).is_dir()]

def find_file(kind, *parts):
    """First existing config/<kind>/<parts...> across the layers, or None"""
    for d in layers(kind):
        if d.joinpath(*parts).exists():
            return d.joinpath(*parts)
    return None

def find_config(kind):
    """The directory of a kind in the highest layer that has it (use `layers` to see them all)"""
    ls = layers(kind)
    return ls[0] if ls else PATH_CONFIG / kind

# ---- Domains
# A domain is a folder of definitions, config/<domain>/<category>/[<group>/]<name>.json. The file's "base" is its category,
# which picks the loader. Names are unique within a domain, so a reference never needs the folders.
# New category: register a loader. New domain: `register_domain`. `ls`, `find` and `load` then work on it.

class Domain:
    def __init__(self, name, loaders=None):
        self.name = name
        self.loaders = {k.lower(): v for k, v in (loaders or {}).items()} # category -> loader(name, obj)

    def register(self, category, loader=None):
        """Add a category's loader: `domain.register("psk", fn)`, or as a decorator `@domain.register("psk")`"""
        def add(fn):
            self.loaders[category.lower()] = fn
            return fn
        return add(loader) if loader is not None else add

_DOMAINS: dict = {}

def register_domain(name, loaders=None) -> Domain:
    """Make (or get, adding `loaders` to it) the domain `name`"""
    d = _DOMAINS.setdefault(name, Domain(name))
    for k, v in (loaders or {}).items():
        d.register(k, v)
    return d

def domains():
    return sorted(_DOMAINS)

def _roots(domain, path):
    return [Path(path)] if path else layers(domain)

def ls(domain, path=None):
    """Definitions of a domain: {"<category>[/<group>]": [names]}, merged across layers (files straight in the domain are under the domain's name)"""
    out: dict = {}
    for root in _roots(domain, path):
        for f in sorted(root.rglob("*.json")):
            rel = f.parent.relative_to(root).as_posix()
            key = domain if rel == "." else rel
            out[key] = sorted(set(out.get(key, [])) | {f.stem})
    return dict(sorted(out.items()))

_log = logging.getLogger("rpps.config")
_WARNED: set = set()

def _deprecated(old, new):
    if old not in _WARNED:
        _WARNED.add(old)
        _log.warning("config name %r is deprecated, use %r", old, new)

def _files(base):
    return sorted(base.rglob("*.json")) if base.is_dir() else []

def locate(domain, ref, category=None, path=None):
    """Find a definition by reference: (file, parts) or (None, None). `parts` is the file's path below the domain, without ".json",
    e.g. ("block", "rs", "204_188"): the category, the group folders, the name.

    ref is the path below the category ("rs/204_188", case-insensitive); without folders it's a bare name, which must match one file.
    Tried in each layer, user layers first: the exact path, a bare name, an "aliases" entry of a definition, and the old "family.name" form.
    category: only look under that folder"""
    ref_l = ref.lower().strip("/")
    for root in _roots(domain, path):
        base = root / category if category else root
        files = _files(base)
        rel = {f.relative_to(base).with_suffix("").as_posix().lower(): f for f in files}
        hit = rel.get(ref_l)
        if hit is None and "/" not in ref_l:
            same = [f for f in files if f.stem.lower() == ref_l]
            if len(same) > 1:
                raise ConfigError(f"{domain}/{ref} is ambiguous, use one of {sorted(f.relative_to(base).with_suffix('').as_posix() for f in same)}")
            hit = same[0] if same else None
        if hit is None:
            for f in files:
                if ref_l in [str(a).lower() for a in read(f).get("aliases", [])]:
                    hit = f
                    _deprecated(ref, f.relative_to(base).with_suffix("").as_posix())
                    break
        if hit is None and "." in ref_l and "/" not in ref_l: # rs.204_188 -> rs/204_188
            new = ref_l.replace(".", "/", 1)
            hit = rel.get(new)
            if hit is not None:
                _deprecated(ref, new)
        if hit is not None:
            return hit, hit.relative_to(root).with_suffix("").parts
    return None, None

def find(domain, ref, category=None, path=None):
    """File of the definition `ref` in the first layer that has it, or None (see `locate`)"""
    return locate(domain, ref, category, path)[0]

def read(path):
    """The definition in a file"""
    return json.loads(Path(path).read_text())

def load(domain, name, category=None, path=None, params=None, example=False):
    """Load `name` of a domain: find the file, then hand its object to the loader of its "base" category.
    params: values for the definition's parameters (see `resolve`). example: fill the others with their examples"""
    file = find(domain, name, category, path)
    if file is None:
        raise FileNotFoundError(f"{domain}/{name}: not found, have {ls(domain, path)}")
    obj = read(file)
    if example:
        params = {**examples(obj), **(params or {})}
    return load_obj(domain, name, obj, params)

def load_obj(domain, name, obj, params=None):
    """Build a definition already read from JSON (or made in code) with the loader of its "base" """
    obj = resolve(obj, params, f"{domain}/{name}")
    d = _DOMAINS.get(domain)
    base = str(obj.get("base", "")).lower()
    if d is None or base not in d.loaders:
        raise ConfigError(f"No {domain} loader for base {obj.get('base')!r} in {name}, have {sorted(d.loaders) if d else []}")
    return d.loaders[base](name, obj)

# ---- Parameters given when a definition is loaded

def declared(obj):
    """{name: default (None: required)} of a definition's "params": {"pci": {"default": 0, "example": 7, "comment": ...}} or {"pci": 0}"""
    out = {}
    for k, v in (obj.get("params") or {}).items():
        out[k] = v.get("default") if isinstance(v, dict) else v
    return out

def examples(obj):
    """Example values for a definition's parameters: each one's "example", else its default (tests and docs load with these)"""
    return {k: (v.get("example", v.get("default")) if isinstance(v, dict) else v) for k, v in (obj.get("params") or {}).items()}

def resolve(obj, params=None, name="definition", strict=True):
    """The definition with its parameters filled in

    A definition declares what it takes under "params" and uses them anywhere in its values:
        {"param": "channel"}                                     the value given (or its default)
        {"expr": "((ns//2 + 1)*(2*pci + 1) << 9) + pci"}        integer arithmetic over the parameters
        {"each": "(i*181) % 196", "n": 196}                       a list: the expression for i = 0 .. n-1
    ("each" lists are built even in a definition without parameters.)
    An expression may use + - * // % ** << >> & | ^ ~, comparisons, `a if c else b`, and/or, list literals and
    indexing, min / max / abs / len / range-free names only: no calls beyond those, no attributes.
    strict: a parameter the definition does not declare is an error (a typo); a chain passes its parameters on
    with strict=False, so each stage takes the ones it knows.
    """
    params = dict(params or {})
    decl = declared(obj)
    unknown = [k for k in params if k not in decl]
    if unknown and strict:
        raise ConfigError(f"{name}: unknown parameter(s) {unknown}; it takes {sorted(decl) or 'none'}")
    values = {k: params.get(k, v) for k, v in decl.items()}
    missing = [k for k, v in values.items() if v is None]
    if missing and _uses_params(obj):
        raise ConfigError(f"{name}: parameter(s) {missing} needed, e.g. load(..., {missing[0]}=...)")
    return _subst(obj, values, name) if decl or _uses_params(obj) else obj

def _uses_params(v):
    if isinstance(v, dict):
        return "param" in v or "expr" in v or "each" in v or any(_uses_params(x) for k, x in v.items() if k != "params")
    if isinstance(v, list):
        return any(_uses_params(x) for x in v)
    return False

def _subst(v, values, name):
    if isinstance(v, dict):
        if set(v) == {"param"}:
            return values[v["param"]]
        if set(v) <= {"expr", "comment"} and "expr" in v:
            return evaluate(v["expr"], values, name)
        if set(v) <= {"each", "n", "comment"} and "each" in v: # A list: the expression at i = 0 .. n-1
            n = _subst(v["n"], values, name)
            return [evaluate(v["each"], {**values, "i": i}, name) for i in range(int(n))]
        return {k: (x if k == "params" else _subst(x, values, name)) for k, x in v.items()}
    if isinstance(v, list):
        return [_subst(x, values, name) for x in v]
    return v

_FUNCS = {"min": min, "max": max, "abs": abs, "len": len}

def evaluate(expr, values, name="expression"):
    """Value of a small integer expression over `values` (see `resolve`): parsed, never handed to eval()"""
    import ast
    import operator as op
    bins = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.FloorDiv: op.floordiv, ast.Mod: op.mod, ast.Pow: op.pow,
            ast.LShift: op.lshift, ast.RShift: op.rshift, ast.BitAnd: op.and_, ast.BitOr: op.or_, ast.BitXor: op.xor, ast.Div: op.truediv}
    cmps = {ast.Eq: op.eq, ast.NotEq: op.ne, ast.Lt: op.lt, ast.LtE: op.le, ast.Gt: op.gt, ast.GtE: op.ge}
    def ev(n):
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float, bool)):
            return n.value
        if isinstance(n, ast.Name):
            if n.id in values:
                return values[n.id]
            raise ConfigError(f"{name}: {expr!r} uses {n.id!r}, which is not a parameter ({sorted(values)})")
        if isinstance(n, ast.BinOp) and type(n.op) in bins:
            return bins[type(n.op)](ev(n.left), ev(n.right))
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.USub, ast.UAdd, ast.Invert, ast.Not)):
            x = ev(n.operand)
            return {ast.USub: lambda: -x, ast.UAdd: lambda: +x, ast.Invert: lambda: ~x, ast.Not: lambda: not x}[type(n.op)]()
        if isinstance(n, ast.Compare) and all(type(o) in cmps for o in n.ops):
            left = ev(n.left)
            for o, c in zip(n.ops, n.comparators):
                right = ev(c)
                if not cmps[type(o)](left, right):
                    return False
                left = right
            return True
        if isinstance(n, ast.BoolOp):
            vals = [ev(x) for x in n.values]
            if isinstance(n.op, ast.And):
                return all(vals) and vals[-1]
            return next((x for x in vals if x), vals[-1])
        if isinstance(n, ast.IfExp):
            return ev(n.body) if ev(n.test) else ev(n.orelse)
        if isinstance(n, (ast.List, ast.Tuple)):
            return [ev(x) for x in n.elts]
        if isinstance(n, ast.Subscript):
            return ev(n.value)[ev(n.slice)]
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in _FUNCS and not n.keywords:
            return _FUNCS[n.func.id](*[ev(a) for a in n.args])
        raise ConfigError(f"{name}: {expr!r}: {type(n).__name__} is not allowed in a definition expression")
    try:
        tree = ast.parse(str(expr), mode="eval")
    except SyntaxError as e:
        raise ConfigError(f"{name}: can't parse {expr!r}: {e.msg}") from None
    return ev(tree)

# ---- Writing definitions

_FIRST = ("base", "type", "comment", "ref", "aliases") # Header keys, in this order

def format_json(v, ind=0):
    """The style of every file in config/: header keys first, a row of numbers on one line, one nested row per line"""
    pad = "  " * ind
    flat = lambda x: not isinstance(x, (list, dict))
    if isinstance(v, dict):
        if not v:
            return "{}"
        keys = [k for k in _FIRST if k in v] + [k for k in v if k not in _FIRST] if ind == 0 else list(v)
        return "{\n" + ",\n".join(f'{pad}  {json.dumps(k)}: {format_json(v[k], ind + 1)}' for k in keys) + f"\n{pad}}}"
    if isinstance(v, list):
        if all(flat(x) for x in v):
            return json.dumps(v)
        if all(isinstance(x, dict) for x in v) and all(all(flat(y) for y in x.values()) for x in v):
            return "[\n" + ",\n".join(f"{pad}  {json.dumps(x)}" for x in v) + f"\n{pad}]"
        return "[\n" + ",\n".join(f"{pad}  {format_json(x, ind + 1)}" for x in v) + f"\n{pad}]"
    return json.dumps(v)

def save(domain, category, ref, obj, dest=None, overwrite=False):
    """Write a definition to <dest>/<domain>/<category>/<ref>.json and return the file

    ref: the name, with any group folders ("taps/my_filter"): lowercase letters, digits, "_", "-" and "." in each part, never absolute or "..".
    dest: a config directory laid out like config/; default the first one added with `add_config_path`, else the shipped config.
    Refuses to replace an existing file unless `overwrite`"""
    import re
    ref = str(ref).lower()
    if not ref or not all(re.fullmatch(r"[a-z0-9_.\-]+", part) and not part.startswith(".") for part in ref.split("/")):
        raise ValueError(f"{ref!r} is not a definition name: use lowercase letters, digits, _ - . in folders and name")
    root = Path(dest) if dest else (_EXTRA[0] if _EXTRA else PATH_CONFIG)
    file = root / domain / category / f"{ref}.json"
    if file.exists() and not overwrite:
        raise FileExistsError(f"{file} exists (overwrite=True replaces it)")
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(format_json(obj) + "\n")
    return file


LOG_CODING = logging.INFO
LOG_CODING_TYPES = logging.WARNING

LOG_SCRAM = logging.INFO

LOG_MOD = logging.INFO
LOG_MOD_CONST = logging.WARNING
