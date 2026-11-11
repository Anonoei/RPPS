"""Link definitions, one resource each: config/crc, config/frame, config/header/link; `rp.link.load("crc", "crc32_mpeg2")`

    crc    {"base": "crc", "type": "crc", "width": 16, "poly": "0x1021", "init": "0x0", "xorout": "0x0",
            "refin": false, "refout": false,       the reveng catalogue fields (numbers or hex strings)
            "check": "0x29b1"}                      optional: the CRC of b"123456789", verified on load
           width is 1 to 64 bits; encode / decode / syndrome work on bit streams of any length (see `crc`)
    frame  {"base": "frame", "type": "hdlc", "flag": "0x7E", "stuff": 5, "nrzi": false,
            "fcs": "crc16_x25",                     the name of a crc definition
            "min_bytes": 4}
    header {"base": "header", "type": "link", "fields": [...], "crc": ..., "payload": "..."}   fields, CRC and payload length of a
            frame's header, driven and published through a profile (see rpps.link.header)
"""
from pathlib import Path

from .. import _config
from .encap import Encap, CRC, Frame
from .header import Header

KINDS = ("crc", "frame", "header")

DOMAINS = {"crc": _config.register_domain("crc", {"crc": CRC.load}), "frame": _config.register_domain("frame", {"frame": Frame.load})}

def root() -> Path:
    """Directory holding the encapsulation definitions"""
    return _config.find_config("crc")

def ls(path=None):
    """List available definitions: {category: [names]}"""
    return {k: sorted(n for names in _config.ls(k, path).items() if k != "header" or names[0] == "link" for n in names[1])
            for k in KINDS}

def load(kind: str, ref: str, path=None, **params):
    """Load config/<kind>/<name>.json (a link header: config/header/link/<name>.json), or from `path`/<kind>/. kind: crc | frame | header"""
    if kind not in KINDS:
        raise ValueError(f"kind is one of {KINDS}, not {kind!r}")
    file, parts = _config.locate(kind, ref, "link" if kind == "header" else None, path)
    if file is None:
        raise FileNotFoundError(f"{kind}/{ref}: not found, have {_config.ls(kind, path)}")
    obj = _config.resolve(_config.read(file), params, f"{kind}/{ref}")
    return _config.load_obj(kind, parts[-1], obj)
