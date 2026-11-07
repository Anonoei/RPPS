"""Encapsulation de-serialization: config/encap/<kind>/<name>.json, kind = crc | frame"""
from pathlib import Path
import json

from .. import _config


KINDS = ("crc", "frame")

def root() -> Path:
    """Directory holding the encapsulation definitions"""
    return _config.find_config("encap")

def ls(path=None):
    """List available definitions: {kind: [names]}"""
    bases = [Path(path)] if path else _config.layers("encap")
    out = {}
    for base in bases:
        for f in sorted(base.iterdir()):
            if f.is_dir():
                out[f.name] = sorted(set(out.get(f.name, [])) | {x.stem for x in f.iterdir() if x.suffix == ".json"})
    return out

def load(kind: str, name: str, path=None):
    """Load config/encap/<kind>/<name>.json (or from `path`/<kind>/)"""
    from .encap import Encap
    file = Path(path) / kind / f"{name.lower()}.json" if path else _config.find_file("encap", kind, f"{name.lower()}.json")
    if file is None or not file.exists():
        raise FileNotFoundError(f"encap/{kind}/{name}: not found, have {ls()}")
    obj = json.loads(file.read_text())
    return Encap.load(name.split(".")[0], obj)
