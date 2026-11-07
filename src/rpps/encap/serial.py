"""Encapsulation de-serialization: config/encap/<category>/<name>.json, category = crc | frame"""
from pathlib import Path

from .. import _config
from .encap import Encap, CRC, Frame

KINDS = ("crc", "frame")

DOMAIN = _config.register_domain("encap", {"crc": CRC.load, "frame": Frame.load, "encap": Encap.load}) # "encap": the type picks

def root() -> Path:
    """Directory holding the encapsulation definitions"""
    return _config.find_config("encap")

def ls(path=None):
    """List available definitions: {category: [names]}"""
    return _config.ls("encap", path)

def load(kind: str, ref: str, path=None):
    """Load config/encap/<kind>/<name>.json (or from `path`/<kind>/). kind: crc | frame"""
    file, parts = _config.locate("encap", ref, kind, path)
    if file is None:
        raise FileNotFoundError(f"encap/{kind}/{ref}: not found, have {ls(path)}")
    obj = _config.read(file)
    return DOMAIN.loaders[str(obj.get("base", kind)).lower()](parts[-1], obj)
