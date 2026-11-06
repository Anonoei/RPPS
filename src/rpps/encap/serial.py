"""Encapsulation de-serialization: config/encap/<kind>/<name>.json, kind = crc | frame"""
from pathlib import Path
import json

from pyboiler.config import config

KINDS = ("crc", "frame")

def root() -> Path:
    """Directory holding the encapsulation definitions"""
    return config().PATH_CONFIG / "encap"  # type: ignore

def ls(path=None):
    """List available definitions: {kind: [names]}"""
    base = Path(path) if path else root()
    return {f.name: sorted(x.stem for x in f.iterdir() if x.suffix == ".json") for f in sorted(base.iterdir()) if f.is_dir()}

def load(kind: str, name: str, path=None):
    """Load config/encap/<kind>/<name>.json (or from `path`/<kind>/)"""
    from .encap import Encap
    base = Path(path) if path else root()
    obj = json.loads((base / kind / f"{name.lower()}.json").read_text())
    return Encap.load(name.split(".")[0], obj)
