import copy
import os
import re
import sys
import functools
import json
import logging
from pathlib import Path

from .errors import ConfigError, ConfigWarning

BYTE_ORDER = ">" if sys.byteorder == "big" else "<"

# Definitions (mod/coding/scram/encap/chain) ship with the package. A directory in $RPPS_CONFIG, or
# one added with `add_config_path`, is searched first, so a project can add or override definitions.
# In a checkout that's config/ next to src/; an installed package carries a copy next to its modules (setup.py puts it there)
# $RPPS_CONFIG_SHIPPED=0 (`use_shipped(False)`) leaves the shipped ones out: only the user's directories are searched.
# $RPPS_CONFIG_TAGS=dvb-s2,lte (`select_tags`) narrows what the blind tools try (`ls(..., blind=True)`) to those tags.
_HERE = Path(__file__).resolve().parent
PATH_CONFIG = _HERE.parents[1] / "config" if (_HERE.parents[1] / "config").is_dir() else _HERE / "config"
_EXTRA: list = [Path(p) for p in os.environ.get("RPPS_CONFIG", "").split(os.pathsep) if p]
_SHIPPED = os.environ.get("RPPS_CONFIG_SHIPPED", "1").strip().lower() not in ("0", "false", "no", "off")
_SELECTED = tuple(t.strip().lower() for t in os.environ.get("RPPS_CONFIG_TAGS", "").split(",") if t.strip()) or None
_CHECKED = False # The selected tags checked against tags() (an environment variable's are checked at first use)

def add_config_path(path):
    """Search `path` (laid out like the shipped config/) before the packaged definitions"""
    _EXTRA.insert(0, Path(path))

def use_shipped(on: bool = True):
    """Search the definitions shipped with RPPS (on, the default), or only the directories added with `add_config_path` /
    $RPPS_CONFIG (off; $RPPS_CONFIG_SHIPPED=0 at start). Off, a shipped name is not found and the blind tools try only
    the user's definitions; code that loads a shipped definition by name (the NR grid's codes, a profile's pulses) needs
    the user's directory to hold it"""
    global _SHIPPED
    _SHIPPED = bool(on)

def select_tags(tags=None):
    """What the blind tools try (identify, survey, solve, the catalogue: `ls(..., blind=True)`): only the definitions
    carrying one of `tags` (["dvb-s2", "lte"]), plus the untagged generic ones (a CRC-32, a K=7 code). None: every one
    ($RPPS_CONFIG_TAGS, comma separated, at start). A definition loads by name whichever tags are selected"""
    global _SELECTED, _CHECKED
    sel = None if tags is None else tuple(dict.fromkeys(str(t).strip().lower() for t in ([tags] if isinstance(tags, str) else tags)))
    if sel is not None:
        _check_tags(sel)
    _SELECTED, _CHECKED = sel or None, True

def selected_tags():
    """The tags `select_tags` chose (a tuple), or None for every one"""
    if _SELECTED is not None and not _CHECKED:
        select_tags(_SELECTED)
    return _SELECTED

def _check_tags(sel):
    unknown = sorted(set(sel) - set(tags()))
    if unknown:
        raise ConfigError(f"select_tags: no tag {unknown} ($RPPS_CONFIG_TAGS or rp.select_tags), have {sorted(tags())}")

def restore(saved):
    """Search as a `state()` taken in another process does (a spawned worker starts from the environment alone)"""
    global _SHIPPED, _SELECTED, _CHECKED
    extra, _SHIPPED, _SELECTED = saved
    _EXTRA[:] = extra
    _CHECKED = False

def roots():
    """The config directories searched, highest priority first: the user's, then the shipped one (unless `use_shipped(False)`)"""
    return [*_EXTRA, *([PATH_CONFIG] if _SHIPPED else [])]

def state():
    """What the definitions found and tried depend on: the directories, the shipped switch, the selected tags (a cache key)"""
    return (tuple(_EXTRA), _SHIPPED, _SELECTED)

def layers(kind):
    """Directories holding definitions of a kind ("constellation", "fec", ...), highest priority first"""
    return [d / kind for d in roots() if (d / kind).is_dir()]

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

EXAMPLE = "example" # The tag of a made-up definition (a demo, a docs example): loadable by name, never a blind candidate

def ls(domain, path=None, all=False, blind=False):
    """Definitions of a domain: {"<category>[/<group>]": [names]}, merged across layers (files straight in the domain are under the domain's name).
    all: also the abstract parents (definitions only children load).
    blind: only what the blind tools (identify, survey, solve, the catalogue) try: not the definitions tagged "example"
    (made up to show a feature, no system uses them), and under `select_tags` only the tagged ones carrying a selected
    tag (untagged generic ones stay: QPSK, whose DVB-S2 map is tagged, is still a generic set)"""
    roots_ = _roots(domain, path)
    key = (domain, str(path) if path else None, all, blind, state(), selected_tags() if blind else None,
           tuple(_fingerprint(r) for r in roots_))
    hit = _LS.get(key)
    if hit is not None:
        return {k: list(v) for k, v in hit.items()}
    out = _ls(domain, roots_, all, blind)
    if len(_LS) > 256:
        _LS.clear()
    _LS[key] = out
    return {k: list(v) for k, v in out.items()}

_LS: dict = {} # ls's answers, by its arguments, the config state and the files' fingerprint

def _fingerprint(root):
    """Every .json below root with its modification time (one scandir walk: no parsing): what ls's answer depends on"""
    out = []
    stack = [str(root)]
    while stack:
        d = stack.pop()
        try:
            it = os.scandir(d)
        except OSError:
            continue
        with it:
            for e in it:
                if e.is_dir(follow_symlinks=False):
                    stack.append(e.path)
                elif e.name.endswith(".json"):
                    try:
                        out.append((e.path, e.stat().st_mtime_ns))
                    except OSError:
                        pass
    return tuple(sorted(out))

def _ls(domain, roots_, all, blind):
    out: dict = {}
    seen = set() # (a name a higher layer has decides, as `locate` does)
    sel = set(selected_tags() or ()) if blind else set()
    for root in roots_:
        for f in _files(root):
            rel = f.parent.relative_to(root).as_posix()
            key = domain if rel == "." else rel
            if (key, f.stem) in seen:
                continue
            seen.add((key, f.stem))
            if not all and _head(f).get("abstract"):
                continue
            if blind:
                t = file_tags(f)
                if EXAMPLE in t or (sel and t and not sel & {*t, *(x for _, ts in map_tags(f) for x in ts)}):
                    continue
            out.setdefault(key, set()).add(f.stem)
    return {k: sorted(v) for k, v in sorted(out.items())}


def _files(base):
    """The definitions below `base`: its files, and the variants their parents hold, each as the file it would be
    (<the parent's folder>/<name>.json, which `read_raw` knows; see `variants`)"""
    if not base.is_dir():
        return []
    out = []
    for f in sorted(base.rglob("*.json")):
        out.append(f)
        out += [f.parent / f"{n}.json" for n in _head(f).get("variants", ())]
    return out

_INDEX = {} # str(directory) -> its definitions by path and by name
@functools.lru_cache(maxsize=4096)
def _real(path) -> Path:
    """Path(path).resolve(), once a path: resolving is a stat a part, and loading a step reads the same files again"""
    return Path(path).resolve()

_LOCATED: dict = {} # str(resolved file) -> (domain, parts): where `locate` found it, for files below a `path` given to it

def _index(base, fresh=False):
    """The definitions below `base`, scanned once and again when `fresh` (a miss: a file may have been written since)"""
    key = str(base)
    if fresh or key not in _INDEX:
        files = _files(base)
        names = {}
        for f in files:
            names.setdefault(f.stem.lower(), []).append(f)
        _INDEX[key] = {"files": files, "names": names,
                       "rel": {f.relative_to(base).with_suffix("").as_posix().lower(): f for f in files}}
    return _INDEX[key]

def _lookup(domain, ref, ref_l, base, fresh):
    idx = _index(base, fresh)
    hit = idx["rel"].get(ref_l)
    if hit is None and "/" not in ref_l:
        same = idx["names"].get(ref_l, [])
        if len(same) > 1:
            raise ConfigError(f"{domain}/{ref} is ambiguous, use one of {sorted(f.relative_to(base).with_suffix('').as_posix() for f in same)}")
        hit = same[0] if same else None
    return hit

def locate(domain, ref, category=None, path=None):
    """Find a definition by reference: (file, parts) or (None, None). `parts` is the file's path below the domain, without ".json",
    e.g. ("block", "rs", "204_188"): the category, the group folders, the name.

    ref is the path below the category ("rs/204_188", case-insensitive); without folders it's a bare name, which must match one file.
    Tried in each layer, user layers first: the exact path, then a bare name. (Which standard uses a definition is its
    "tags", not another name: `tagged("dvb-s")`.)
    category: only look under that folder"""
    ref_l = ref.lower().strip("/")
    for root in _roots(domain, path):
        base = root / category if category and not (category == domain and not (root / category).is_dir()) else root
        hit = _lookup(domain, ref, ref_l, base, False)
        if hit is None or not _exists(hit): # Not there when the folder was scanned, or gone since
            hit = _lookup(domain, ref, ref_l, base, True)
        if hit is not None:
            parts = hit.relative_to(root).with_suffix("").parts
            _LOCATED[str(_real(str(hit)))] = (domain, parts)
            return hit, parts
    return None, None

def find(domain, ref, category=None, path=None):
    """File of the definition `ref` in the first layer that has it, or None (see `locate`)"""
    return locate(domain, ref, category, path)[0]

def read_raw(path):
    """The definition in a file as written (no "extends" applied): what `fmt` formats and `save` writes. A variant (a row
    of its parent's "variants") as the child file it stands for: {"extends": <the parent>, <its columns>}"""
    v = _variant(path)
    if v is not None:
        return json.loads(json.dumps(v[1]))
    return json.loads(Path(path).read_text())

# ---- Variants: a parent's children written as rows of it, each a definition named as a file next to the parent would be

_VARIANTS: dict = {} # str(file a variant stands for, as `_files` made it) -> (its parent's file, the child it is)
_KIDS: dict = {} # str(parent file) -> the keys of its variants in _VARIANTS

def _variant(f):
    """(parent file, child) of the variant `f` stands for, its parent read again if it changed; None for any other file"""
    v = _VARIANTS.get(str(f))
    if v is None or not v[0].exists():
        return None
    _head(v[0])
    return _VARIANTS.get(str(f))

def _exists(f):
    """A definition file is there, or the parent holding the variant it stands for is"""
    return _variant(f) is not None or Path(f).exists()

def variants(f, raw):
    """{name: child} of a parent file's "variants" as written: one child (what a file extending the parent would hold)
    a row. "columns" name the keys ("name" first: the variant's file name next to the parent's), a dotted one a key in a
    dict ("params.gamma"); each row gives their values, as a child file would (null deletes a parent's key)"""
    v = raw.get("variants")
    if v is None:
        return {}
    if (not isinstance(v, dict) or set(v) - {"columns", "rows", "comment"} or not isinstance(v.get("columns"), list)
            or not v["columns"] or v["columns"][0] != "name" or not isinstance(v.get("rows"), list)):
        raise ConfigError(f'{f}: "variants" takes {{"columns": ["name", ...], "rows": [[...], ...]}}')
    cols, out = v["columns"], {}
    for c in cols[1:]:
        if c in ("extends", "abstract", "base", "variants") or not all(c.split(".")):
            raise ConfigError(f"{f}: variants: {c!r} is not a column a variant can set")
    for row in v["rows"]:
        if not isinstance(row, list) or len(row) != len(cols):
            raise ConfigError(f"{f}: variants: the row {row!r} does not have the {len(cols)} values of {cols}")
        name = str(row[0]).lower()
        if not re.fullmatch(r"[a-z0-9_.\-]+", name) or name.startswith("."):
            raise ConfigError(f"{f}: variants: {row[0]!r} is not a name (lowercase letters, digits, _ - .)")
        if name in out or name == Path(f).stem.lower():
            raise ConfigError(f"{f}: variants: {name!r} twice")
        child = {"extends": Path(f).stem}
        for c, x in zip(cols[1:], row[1:]):
            *head, last = c.split(".")
            at = child
            for h in head:
                at = at.setdefault(h, {})
                if not isinstance(at, dict):
                    raise ConfigError(f"{f}: variants: {c!r} goes inside {h!r}, which another column sets")
            at[last] = x
        out[name] = child
    return out

def read(path):
    """The definition in a file, its "extends" applied (see `expand`). An abstract definition (a parent only children load) is an error"""
    obj = expand(path)
    if obj.get("abstract"):
        raise ConfigError(f"{path}: abstract (a parent for other definitions to extend), load one of its children")
    return obj

# ---- Inheritance: a definition that "extends" another is that one with its own keys changed

_NOT_INHERITED = ("comment", "ref", "abstract", "extends", "variants")

_HEAD: dict = {} # str(file) -> (mtime, the keys `ls` and `tagged` need)

def _head(f):
    """{"abstract", "tags", "extends", "variants": [names]} of a file as written, cached until it changes (a variant's: of
    the child it is)"""
    f = Path(f)
    key = str(f)
    if key in _VARIANTS:
        v = _variant(f)
        if v is not None:
            return {k: v[1][k] for k in ("tags", "extends") if k in v[1]}
    mt = f.stat().st_mtime_ns
    hit = _HEAD.get(key)
    if hit is None or hit[0] != mt:
        raw = json.loads(f.read_text())
        kids = variants(f, raw)
        for k in _KIDS.pop(key, ()):
            _VARIANTS.pop(k, None)
        for name, child in kids.items():
            vf = f.parent / f"{name}.json"
            if vf.exists():
                raise ConfigError(f"{f}: the variant {name!r} is also a file, {vf}")
            _VARIANTS[str(vf)] = (f, child)
        _KIDS[key] = [str(f.parent / f"{name}.json") for name in kids]
        maps = {"maps": [[m.get("comment", ""), list(m["tags"])] for m in raw["maps"] if isinstance(m, dict) and m.get("tags")]} \
            if isinstance(raw.get("maps"), list) else {} # (a constellation's maps that carry tags of their own)
        hit = (mt, {k: raw[k] for k in ("abstract", "tags", "extends") if k in raw} | ({"variants": list(kids)} if kids else {}) | maps)
        _HEAD[key] = hit
    return hit[1]

def place(path):
    """(domain, parts below the domain without ".json") of a definition file, or (None, None) outside every config layer"""
    path = _real(str(path))
    for root in (*_EXTRA, PATH_CONFIG):
        try:
            rel = path.relative_to(_real(str(root)))
        except ValueError:
            continue
        if len(rel.parts) > 1:
            return rel.parts[0], rel.with_suffix("").parts[1:]
    return _LOCATED.get(str(path), (None, None)) # (below a `path` given to `locate`)

def base_of(domain, parts):
    """The loader ("base") of the definition at `parts` below `domain`: the folder it is in says it. The first folder below
    the resource when that is one of its loaders (fec/block, fec/conv, filter/fir, plan/builder), else the resource itself"""
    d = _DOMAINS.get(domain)
    if d is None:
        raise ConfigError(f"{domain}/{'/'.join(parts)}: no resource {domain!r}, have {domains()}")
    if len(parts) > 1 and parts[0].lower() in d.loaders:
        return parts[0].lower()
    if domain in d.loaders:
        return domain
    raise ConfigError(f"{domain}/{'/'.join(parts)}: a {domain} definition is in one of the folders {sorted(d.loaders)}")

def file_base(path):
    """The loader ("base") of a definition file, from where it is (`base_of`)"""
    domain, parts = place(path)
    if domain is None:
        raise ConfigError(f"{path}: not in a config folder (rp.add_config_path), so the folder can't say what it is")
    return base_of(domain, parts)

def _parent(path, ref):
    """The file a definition at `path` extends: `ref` below the definition's category first (fec/block: "rs/p25"), then below
    its domain. A variant's is the parent holding it"""
    v = _variant(path)
    if v is not None:
        return v[0]
    domain, parts = place(path)
    if domain is None:
        raise ConfigError(f"{path}: extends {ref!r}, but the file is not in a config folder, so there is nowhere to look")
    first = str(ref).split("/")[0]
    if "/" in str(ref) and first in _DOMAINS and first != domain:
        raise ConfigError(f"{domain}/{'/'.join(parts)}: extends {ref!r} in another resource; a definition extends one of its own kind")
    hit = None
    if len(parts) > 1:
        hit, _ = locate(domain, ref, parts[0])
    if hit is None:
        hit, _ = locate(domain, ref)
    if hit is None:
        raise ConfigError(f"{domain}/{'/'.join(parts)}: extends {ref!r}, which is not a {domain} definition")
    return hit

def merge(parent, child):
    """A child definition over its parent: dicts merge key by key, lists and values replace, null deletes. comment and ref
    are the child's own; tags add to the parent's; params merge by name (a bare value pins the parent's default); "steps"
    given as a dict patches the parent's steps by id (see `patch_steps`)"""
    out = {k: v for k, v in parent.items() if k not in _NOT_INHERITED}
    for k, v in child.items():
        if k == "extends":
            continue
        if v is None:
            out.pop(k, None)
        elif k == "tags":
            out[k] = list(dict.fromkeys([*out.get(k, []), *v]))
        elif k == "params" and isinstance(out.get(k), dict) and isinstance(v, dict):
            ps = dict(out[k])
            for name, pv in v.items():
                old = ps.get(name)
                if pv is None:
                    ps.pop(name, None)
                elif isinstance(old, dict) and not isinstance(pv, dict):
                    ps[name] = {**old, "default": pv}
                elif isinstance(old, dict) and isinstance(pv, dict):
                    ps[name] = {**old, **pv}
                else:
                    ps[name] = pv
            out[k] = ps
        elif k == "steps" and isinstance(v, dict) and isinstance(out.get(k), list):
            out[k] = patch_steps(out[k], v)
        elif isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep(out[k], v)
        else:
            out[k] = v
    return out

STEP_KINDS = ("signal", "sync", "wave", "symbol", "code", "link", "grid", "branch") # The keys that say what a step is

def step_id(step):
    """A profile step's id: its "id", else its kind ({"sync": "timing"}: "timing"), as a profile names it"""
    if "id" in step:
        return str(step["id"])
    return next((str(step[k]) for k in STEP_KINDS if k in step and isinstance(step[k], str)), None)

def patch_steps(steps, patch):
    """A parent's steps with a child's changes, by step id (`step_id`), in the order given:
        "<id>": {...}           merged into that step (as dicts merge: null deletes a key)
        "<id>": null            the step taken out
        "+before:<id>": [...]   steps put in before it; "+after:<id>" after it
        "+start": [...]         steps put first; "+end" last
    An id two steps share is an error: give them "id"s"""
    out = [dict(s) for s in steps]
    def at(sid):
        hits = [i for i, s in enumerate(out) if step_id(s) == sid]
        if len(hits) != 1:
            have = [step_id(s) for s in out]
            raise ConfigError(f"steps: {'no' if not hits else 'more than one'} step {sid!r} to patch (have {have}"
                              + ("; give them \"id\"s)" if hits else ")"))
        return hits[0]
    def new(v, key):
        if not isinstance(v, list) or not all(isinstance(x, dict) for x in v):
            raise ConfigError(f"steps: {key!r} takes a list of steps, not {v!r}")
        return [dict(x) for x in v]
    for key, v in patch.items():
        if key == "comment":
            continue
        if key == "+start":
            out[:0] = new(v, key)
        elif key == "+end":
            out += new(v, key)
        elif key.startswith(("+before:", "+after:")):
            where, _, sid = key[1:].partition(":")
            i = at(sid) + (where == "after")
            out[i:i] = new(v, key)
        elif v is None:
            del out[at(key)]
        elif isinstance(v, dict):
            i = at(key)
            out[i] = _deep(out[i], v)
        else:
            raise ConfigError(f"steps: {key!r} takes a dict (merged into the step) or null (taken out), not {v!r}")
    return out

def _deep(a, b):
    out = dict(a)
    for k, v in b.items():
        if v is None:
            out.pop(k, None)
        elif isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep(out[k], v)
        else:
            out[k] = v
    return out

def join_comments(v, where="definition"):
    """`v` with every "comment" written as a list of lines made one string (the lines joined by spaces), at any depth"""
    if isinstance(v, dict):
        out = {}
        for k, x in v.items():
            if k == "comment" and isinstance(x, list):
                if not all(isinstance(s, str) for s in x):
                    raise ConfigError(f"{where}: a comment is a string or a list of strings, not {x!r}")
                out[k] = " ".join(x)
            else:
                out[k] = join_comments(x, where)
        return out
    if isinstance(v, list):
        return [join_comments(x, where) for x in v]
    return v

_EXPANDED: dict = {} # str(file) -> (what it depends on: the config state and each file's of its chain mtime, the definition)

def _stamp(path):
    """When a definition file last changed (a variant's: its parent's)"""
    v = _variant(path)
    try:
        return Path(v[0] if v is not None else path).stat().st_mtime_ns
    except OSError:
        return None

def expand(path, _chain=()):
    """The definition in a file with its "extends" chain applied (abstract or not), comments one string each, and its
    "base" from the folder it is in (`base_of`). Worked out once and kept until a file of its chain or the config
    directories change (a step built per burst or per frame reads its definitions again): each call gets its own copy"""
    key = str(path)
    hit = _EXPANDED.get(key)
    if hit is not None and hit[0][0] == state() and all(_stamp(f) == t for f, t in hit[0][1]):
        return copy.deepcopy(hit[1])
    files = []
    obj = _expand(Path(path), _chain, files)
    _EXPANDED[key] = ((state(), tuple(files)), obj)
    return copy.deepcopy(obj)

def _expand(path, _chain, files):
    files.append((str(path), _stamp(path)))
    raw = read_raw(path)
    base = file_base(path)
    if "base" in raw:
        raise ConfigError(f"{path}: \"base\" is the folder the file is in ({base!r}), take the key out")
    obj = {"base": base, **join_comments(raw, str(path))}
    ref = obj.get("extends")
    if ref is None:
        return obj
    parent = _parent(path, ref)
    chain = (*_chain, _real(str(path)))
    if _real(str(parent)) in chain:
        raise ConfigError(f"extends loops: {' -> '.join(str(p) for p in (*chain, _real(str(parent))))}")
    try:
        return {**merge(_expand(Path(parent), chain, files), obj), "base": base}
    except ConfigError as e:
        if str(path) in str(e):
            raise
        raise ConfigError(f"{path}: {e}") from None

# ---- Tags: who uses a definition (a standard, a system), from the list in config/tags.json

def tags():
    """{tag: {"comment", "ref"}} of every config layer's tags.json (user layers win). The shipped one is read even with
    `use_shipped(False)`: it names tags, it holds no definitions"""
    out: dict = {}
    for root in (PATH_CONFIG, *reversed(_EXTRA)):
        f = Path(root) / "tags.json"
        if f.is_file():
            out.update(read_raw(f).get("tags", {}))
    return dict(sorted(out.items()))

def file_tags(f, _chain=()):
    """The tags of a definition file, its parents' included"""
    h = _head(f)
    own = list(h.get("tags", []))
    if "extends" not in h:
        return own
    parent = _parent(f, h["extends"])
    if Path(parent).resolve() in _chain:
        return own
    return list(dict.fromkeys([*file_tags(parent, (*_chain, Path(f).resolve())), *own]))

def map_tags(f, _chain=()):
    """[(map name, its tags)] of the maps of a definition file that carry tags of their own (a constellation map a
    standard labels the set by: `{"map": [...], "comment": "DVB-S2", "tags": ["dvb-s2"]}`); a child that writes no
    "maps" has its parent's"""
    h = _head(f)
    if "maps" in h:
        return [tuple(m) for m in h["maps"]]
    if "extends" not in h:
        return []
    parent = _parent(f, h["extends"])
    if Path(parent).resolve() in _chain:
        return []
    return map_tags(parent, (*_chain, Path(f).resolve()))

def tagged(tag=None, all=False):
    """Definitions carrying `tag`: {resource: [refs below it]}. No tag: {tag: {resource: [refs]}} for every tag in use.
    A map that carries a tag of its own lists as "<ref> (<map>)", e.g. "psk/qpsk (DVB-S2)": `rp.symbol.load(ref, map)`.
    all: abstract parents too"""
    found: dict = {}
    for root in roots():
        root = Path(root)
        if not root.is_dir():
            continue
        for d in sorted(p for p in root.iterdir() if p.is_dir()):
            for f in _files(d):
                if not all and _head(f).get("abstract"):
                    continue
                ref = f.relative_to(d).with_suffix("").as_posix()
                own = [(ref, t) for t in file_tags(f)] + [(f"{ref} ({name})", t) for name, ts in map_tags(f) for t in ts]
                for r, t in own:
                    lst = found.setdefault(t, {}).setdefault(d.name, [])
                    if r not in lst:
                        lst.append(r)
    if tag is None:
        return dict(sorted(found.items()))
    t = str(tag).lower()
    if t not in found and t not in tags():
        raise ConfigError(f"No tag {tag!r}, have {sorted(tags())}")
    return found.get(t, {})

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

def load_ref(ref, **params):
    """Any definition by its path below config/: `rp.load("fec/block/rs/204_188")`, `rp.load("crc/crc32_mpeg2")`,
    `rp.load("profile/ieee802154/2450", sps=8)`. The first part is the resource (`rp.domains()` lists them), the rest the
    reference within it (a bare name works when it is unique). params: values for the definition's "params" """
    domain, _, rest = str(ref).strip("/").partition("/")
    if domain not in _DOMAINS:
        raise ConfigError(f"{ref}: no resource {domain!r}, have {domains()}")
    if not rest:
        raise ConfigError(f"{ref}: name a definition below {domain}/, have {ls(domain)}")
    return load(domain, rest, params=params)

# ---- Keys nothing read: a definition's or a step's settings, checked after it is built

class Keys(dict):
    """A dict that records which keys were asked for (`read`), so the keys nothing used can be reported. Going over it
    (iterating, items(), copying) counts as using every key: whatever took the whole dict decides. `read` can be one set
    shared by several Keys (a sync step and the settings merged into its method)"""
    def __init__(self, *args, read=None, **kw):
        super().__init__(*args, **kw)
        self.read = set() if read is None else read
        self.whole = False

    def __getitem__(self, k):
        self.read.add(k)
        return super().__getitem__(k)

    def get(self, k, default=None):
        self.read.add(k)
        return super().get(k, default)

    def __contains__(self, k):
        self.read.add(k)
        return super().__contains__(k)

    def pop(self, k, *default):
        self.read.add(k)
        return super().pop(k, *default)

    def setdefault(self, k, default=None):
        self.read.add(k)
        return super().setdefault(k, default)

    def _all(self):
        self.whole = True

    def __iter__(self):
        self._all()
        return super().__iter__()

    def keys(self):
        self._all()
        return super().keys()

    def items(self):
        self._all()
        return super().items()

    def values(self):
        self._all()
        return super().values()

    def copy(self):
        self._all()
        return dict(super().items())

def unused(obj, exempt=()):
    """The keys of a `Keys` nothing read (none when it was gone over whole, or is a plain dict)"""
    if not isinstance(obj, Keys) or obj.whole:
        return []
    return sorted(k for k in dict.keys(obj) if k not in obj.read and k not in exempt)

def warn_unused(obj, where, exempt=(), stacklevel=3):
    """A ConfigWarning naming the keys of `obj` nothing read"""
    import warnings
    keys = unused(obj, exempt)
    if keys:
        warnings.warn(f"{where}: {', '.join(repr(k) for k in keys)} not used (misspelt, or not a setting of what it names)",
                      ConfigWarning, stacklevel=stacklevel)
    return keys

# What a definition carries for the loaders around it, never read by its own loader ("guess": what a blind tool that
# found it measured)
DEFINITION_KEYS = ("base", "comment", "ref", "extends", "abstract", "tags", "params", "examples", "guess", "variants")

def load_obj(domain, name, obj, params=None):
    """Build a definition already read from JSON (or made in code) with the loader of its "base" """
    obj = resolve(join_comments(obj, f"{domain}/{name}"), params, f"{domain}/{name}")
    d = _DOMAINS.get(domain)
    base = str(obj.get("base", "")).lower()
    if d is None or base not in d.loaders:
        raise ConfigError(f"No {domain} loader for base {obj.get('base')!r} in {name}, have {sorted(d.loaders) if d else []}")
    if domain == "profile": # (a profile checks its own keys and its steps': rpps.profile.Profile)
        return d.loaders[base](name, obj)
    obj = Keys(obj)
    out = d.loaders[base](name, obj)
    warn_unused(obj, f"{domain}/{name}", DEFINITION_KEYS)
    return out

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
        {..., "when": "modulation == '16qam'"}                  in a list: the item only where the expression holds
    ("each" lists and "when" are applied even in a definition without parameters.)
    An expression may use + - * // % ** << >> & | ^ ~, comparisons, `a if c else b`, and/or, number, string and list
    literals, indexing, min / max / abs / len / ceil / floor / round / log2: no calls beyond those, no attributes but a
    choice's fields.

    A param may say its "type" (int, float, str, bool), "min" and "max": a value outside them is an error at load.

    A parameter may list its "choices": the values it takes, each with a bundle of fields one choice sets together
        "modulation": {"default": "qpsk", "choices": {"qpsk": {"constellation": "qpsk", "bits": 2},
                                                      "16qam": {"constellation": "qam/16", "bits": 4}}}
    then {"param": "modulation.constellation"}, or `modulation.bits` in an expression, is the chosen bundle's field
    ("choices": [2, 4, 8] only checks the value). A value that is not one of them is an error.
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
    if not (decl or _uses_params(obj)):
        return obj
    typed(obj, values, name)
    values = chosen(obj, values, name)
    out = _subst(obj, values, name)
    if decl: # The filled-in definition says what it was given: `declared` of it gives the values used, not the defaults
        out = dict(out)
        out["params"] = {k: ({**v, "default": values[k]} if isinstance(v, dict) else values[k]) for k, v in obj["params"].items()}
    return out

def _choice(choices, value):
    """The key of `choices` (a list, or a dict of bundles) that is `value`: equal, or its text (JSON keys are strings)"""
    for c in choices:
        if c == value or (isinstance(c, str) and not isinstance(value, str) and c == str(value)):
            return c
    raise KeyError(value)

_TYPES = {"int": (int,), "float": (int, float), "str": (str,), "bool": (bool,)}

def typed(obj, values, name="definition"):
    """Check each param against its "type" (int, float, str, bool), "min" and "max", where it says them: a value of the
    wrong kind is an error when loading, not a failure later in a step"""
    for k, spec in (obj.get("params") or {}).items():
        v = values.get(k)
        if not isinstance(spec, dict) or v is None:
            continue
        t = spec.get("type")
        if t is not None:
            if t not in _TYPES:
                raise ConfigError(f"{name}: param {k}: type is one of {sorted(_TYPES)}, not {t!r}")
            if not isinstance(v, _TYPES[t]) or (t != "bool" and isinstance(v, bool)):
                raise ConfigError(f"{name}: {k}={v!r} is not {'an' if t == 'int' else 'a'} {t}")
        if "min" in spec and v < spec["min"]:
            raise ConfigError(f"{name}: {k}={v!r} is below its minimum {spec['min']}")
        if "max" in spec and v > spec["max"]:
            raise ConfigError(f"{name}: {k}={v!r} is above its maximum {spec['max']}")

def chosen(obj, values, name="definition"):
    """The parameter values with each choice checked and its bundle's fields added as "<param>.<field>" """
    out = dict(values)
    for k, spec in (obj.get("params") or {}).items():
        if not isinstance(spec, dict) or "choices" not in spec or values.get(k) is None:
            continue
        choices = spec["choices"]
        try:
            c = _choice(choices, values[k])
        except KeyError:
            raise ConfigError(f"{name}: {k}={values[k]!r} is not one of its choices {list(choices)}") from None
        if isinstance(choices, dict):
            for f, v in (choices[c] or {}).items():
                if f != "comment":
                    out[f"{k}.{f}"] = v
    return out

def _uses_params(v):
    if isinstance(v, dict):
        return "param" in v or "expr" in v or "each" in v or "when" in v or any(_uses_params(x) for k, x in v.items() if k != "params")
    if isinstance(v, list):
        return any(_uses_params(x) for x in v)
    return False

def _subst(v, values, name):
    if isinstance(v, dict):
        if set(v) == {"param"}:
            if v["param"] not in values:
                raise ConfigError(f"{name}: {v['param']!r} is not a parameter or a choice's field ({sorted(values)})")
            return values[v["param"]]
        if set(v) <= {"expr", "comment"} and "expr" in v:
            return evaluate(v["expr"], values, name)
        if set(v) <= {"each", "n", "comment"} and "each" in v: # A list: the expression at i = 0 .. n-1
            n = _subst(v["n"], values, name)
            return [evaluate(v["each"], {**values, "i": i}, name) for i in range(int(n))]
        return {k: (x if k == "params" else _subst(x, values, name)) for k, x in v.items()}
    if isinstance(v, list):
        out = []
        for x in v:
            if isinstance(x, dict) and "when" in x:
                if not evaluate(x["when"], values, name):
                    continue # Not in this variant
                x = {k: y for k, y in x.items() if k != "when"}
            out.append(_subst(x, values, name))
        return out
    return v

import math as _math
_FUNCS = {"min": min, "max": max, "abs": abs, "len": len, "ceil": _math.ceil, "floor": _math.floor, "round": round,
          "log2": _math.log2}

def evaluate(expr, values, name="expression"):
    """Value of a small integer expression over `values` (see `resolve`): parsed, never handed to eval()"""
    import ast
    import operator as op
    bins = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.FloorDiv: op.floordiv, ast.Mod: op.mod, ast.Pow: op.pow,
            ast.LShift: op.lshift, ast.RShift: op.rshift, ast.BitAnd: op.and_, ast.BitOr: op.or_, ast.BitXor: op.xor, ast.Div: op.truediv}
    cmps = {ast.Eq: op.eq, ast.NotEq: op.ne, ast.Lt: op.lt, ast.LtE: op.le, ast.Gt: op.gt, ast.GtE: op.ge,
            ast.In: lambda a, b: a in b, ast.NotIn: lambda a, b: a not in b}
    def ev(n):
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float, bool, str)):
            return n.value
        if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name): # A choice's field
            k = f"{n.value.id}.{n.attr}"
            if k in values:
                return values[k]
            raise ConfigError(f"{name}: {expr!r} uses {k!r}, which is not a choice's field ({sorted(values)})")
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
        if isinstance(n, ast.BoolOp): # Short-circuit, as Python does: `length > 0 and flags & 16` with no flags field
            v = None
            for x in n.values:
                v = ev(x)
                if (not v) if isinstance(n.op, ast.And) else v:
                    return v
            return v
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
        tree = _parse(str(expr))
    except SyntaxError as e:
        raise ConfigError(f"{name}: can't parse {expr!r}: {e.msg}") from None
    return ev(tree)

@functools.lru_cache(maxsize=1024)
def _parse(expr):
    """The parsed expression, once per text (an "each" list evaluates the same one at every i); the tree is only read"""
    import ast
    return ast.parse(expr, mode="eval")

# ---- Writing definitions

_FIRST = ("base", "extends", "abstract", "type", "comment", "ref", "tags") # Header keys, in this order ("base" only in code: a file's folder says it)

WIDTH = 120 # Columns a line of a definition file fills before what is on it is spread over several

def format_json(v, ind=0, used=0, spread=False):
    """The style of every file in config/: header keys first; whatever fits on its line (`WIDTH`) on one line, a list of
    numbers always on one; table and variant "rows" one a line, however long.
    used: columns already taken on the line this value starts on. spread: never on one line"""
    pad = "  " * ind
    if isinstance(v, list) and not any(isinstance(x, (list, dict)) for x in v):
        return json.dumps(v)
    if not isinstance(v, (list, dict)) or not v:
        return json.dumps(v)
    if ind > 0 and not spread:
        one = json.dumps(v)
        if used + len(one) + 1 <= WIDTH: # (+1: the comma after it)
            return one
    if isinstance(v, dict):
        keys = [k for k in _FIRST if k in v] + [k for k in v if k not in _FIRST] if ind == 0 else list(v)
        lines = []
        for k in keys:
            head = f"{pad}  {json.dumps(k)}: "
            x = v[k]
            if k == "comment" and isinstance(x, (str, list)) and all(isinstance(s, str) for s in x if isinstance(x, list)):
                lines.append(head + _format_comment(" ".join(x) if isinstance(x, list) else x, pad, len(head)))
                continue
            lines.append(head + format_json(x, ind + 1, len(head), spread=k == "rows" and isinstance(x, list)))
        return "{\n" + ",\n".join(lines) + f"\n{pad}}}"
    if spread: # (table rows: one a line, whatever they hold)
        return "[\n" + ",\n".join(f"{pad}  {json.dumps(x)}" for x in v) + f"\n{pad}]"
    return "[\n" + ",\n".join(f"{pad}  {format_json(x, ind + 1, len(pad) + 2)}" for x in v) + f"\n{pad}]"

def comment_lines(text, width):
    """`text` cut at spaces into lines of at most `width` characters as JSON strings (a word longer is a line of its own),
    each line ending a sentence or clause (". ", "; ", ": ") where one ends in its second half: " ".join gives `text` back"""
    words, lines, cur = text.split(" "), [], []
    size = lambda ws: len(json.dumps(" ".join(ws)))
    for w in words:
        if cur and size([*cur, w]) > width:
            cut = next((i + 1 for i in range(len(cur) - 1, len(cur)//2 - 1, -1) if cur[i].endswith((".", ";", ":"))), len(cur))
            lines.append(" ".join(cur[:cut]))
            cur = cur[cut:]
        cur.append(w)
    return lines + [" ".join(cur)]

def _format_comment(text, pad, used):
    """A comment on one line when it fits, else a list of lines (`comment_lines`), one a line"""
    if used + len(json.dumps(text)) + 1 <= WIDTH:
        return json.dumps(text)
    lines = comment_lines(text, WIDTH - len(pad) - 4 - 1)
    return "[\n" + ",\n".join(f"{pad}    {json.dumps(s)}" for s in lines) + f"\n{pad}  ]"

def save(domain, category, ref, obj, dest=None, overwrite=False):
    """Write a definition to <dest>/<domain>/<category>/<ref>.json and return the file. Its "base" is not written: the
    folder says it, and must say the same (`base_of`)

    ref: the name, with any group folders ("taps/my_filter"): lowercase letters, digits, "_", "-" and "." in each part, never absolute or "..".
    dest: a config directory laid out like config/; default the first one added with `add_config_path`. Never the shipped
    config/ by default: pass dest=_config.PATH_CONFIG to write there on purpose. Refuses to replace an existing file unless `overwrite`"""
    ref = str(ref).lower()
    if not ref or not all(re.fullmatch(r"[a-z0-9_.\-]+", part) and not part.startswith(".") for part in ref.split("/")):
        raise ValueError(f"{ref!r} is not a definition name: use lowercase letters, digits, _ - . in folders and name")
    if not dest and not _EXTRA:
        raise ConfigError("save: no destination; pass dest=<config dir> or add one with rp.add_config_path(dir)")
    root = Path(dest) if dest else _EXTRA[0]
    parts = (*([category] if category and category != domain else []), *ref.split("/"))
    out = file_obj(domain, parts, obj)
    file = root.joinpath(domain, *parts[:-1], f"{parts[-1]}.json")
    if file.exists() and not overwrite:
        raise FileExistsError(f"{file} exists (overwrite=True replaces it)")
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(format_json(out) + "\n")
    return file

def file_obj(domain, parts, obj):
    """A definition made in code as its file at `parts` below `domain` holds it: without "base", which the folder says,
    and which must be the one it has (`base_of`)"""
    base = base_of(domain, parts)
    if str(obj.get("base", base)).lower() != base:
        raise ConfigError(f"a {obj['base']!r} definition does not go in {domain}/{'/'.join(parts)}, whose folder says {base!r}")
    return {k: v for k, v in obj.items() if k != "base"}


LOG_CODING = logging.INFO
LOG_CODING_TYPES = logging.WARNING

LOG_SCRAM = logging.INFO

LOG_MOD = logging.INFO
LOG_MOD_CONST = logging.WARNING
