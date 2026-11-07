import os
import sys
import logging
from pathlib import Path

BYTE_ORDER = ">" if sys.byteorder == "big" else "<"

# Definitions (mod/coding/scram/encap/chain) ship with the package. A directory in $RPPS_CONFIG, or
# one added with `add_config_path`, is searched first, so a project can add or override definitions.
# In a checkout that's config/ next to src/; an installed package carries a copy next to its modules (setup.py puts it there)
_HERE = Path(__file__).resolve().parent
PATH_CONFIG = _HERE.parents[1] / "config" if (_HERE.parents[1] / "config" / "coding").is_dir() else _HERE / "config"
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

LOG_CODING = logging.INFO
LOG_CODING_TYPES = logging.WARNING

LOG_SCRAM = logging.INFO

LOG_MOD = logging.INFO
LOG_MOD_CONST = logging.WARNING
