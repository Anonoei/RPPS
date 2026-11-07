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

def load(domain, name, category=None, path=None):
    """Load `name` of a domain: find the file, then hand its object to the loader of its "base" category"""
    file = find(domain, name, category, path)
    if file is None:
        raise FileNotFoundError(f"{domain}/{name}: not found, have {ls(domain, path)}")
    return load_obj(domain, name, read(file))

def load_obj(domain, name, obj):
    """Build a definition already read from JSON (or made in code) with the loader of its "base" """
    d = _DOMAINS.get(domain)
    base = str(obj.get("base", "")).lower()
    if d is None or base not in d.loaders:
        raise ConfigError(f"No {domain} loader for base {obj.get('base')!r} in {name}, have {sorted(d.loaders) if d else []}")
    return d.loaders[base](name, obj)

LOG_CODING = logging.INFO
LOG_CODING_TYPES = logging.WARNING

LOG_SCRAM = logging.INFO

LOG_MOD = logging.INFO
LOG_MOD_CONST = logging.WARNING
