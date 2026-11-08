"""Encapsulation definitions: config/encap/<category>/<name>.json, loaded with `rp.encap.load("crc", "crc32_mpeg2")`

    crc    {"base": "crc", "type": "crc", "width": 16, "poly": "0x1021", "init": "0x0", "xorout": "0x0",
            "refin": false, "refout": false,       the reveng catalogue fields (numbers or hex strings)
            "check": "0x29b1"}                      optional: the CRC of b"123456789", verified on load
           width is 1 to 64 bits; encode / decode / syndrome work on bit streams of any length (see `crc`)
    frame  {"base": "frame", "type": "hdlc", "flag": "0x7E", "stuff": 5, "nrzi": false,
            "fcs": "crc16_x25",                     the name of a crc definition
            "min_bytes": 4}
"""
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

def load(kind: str, ref: str, path=None, **params):
    """Load config/encap/<kind>/<name>.json (or from `path`/<kind>/). kind: crc | frame"""
    file, parts = _config.locate("encap", ref, kind, path)
    if file is None:
        raise FileNotFoundError(f"encap/{kind}/{ref}: not found, have {ls(path)}")
    obj = _config.resolve(_config.read(file), params, f"encap/{kind}/{ref}")
    return DOMAIN.loaders[str(obj.get("base", kind)).lower()](parts[-1], obj)
