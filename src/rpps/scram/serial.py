"""Scrambler de-serialization: config/scram/<category>/<name>.json, category = fdt (feedthrough) | adt (additive)"""
from .. import _config

from . import scrambler

DOMAIN = _config.register_domain("scram", {"fdt": scrambler.Feedthrough.load, "adt": scrambler.Additive.load})

def ls():
    """List available scram types: {category: [names]}"""
    return _config.ls("scram")

def init(name: str, obj: dict) -> scrambler.Scram:
    """Initialize scram"""
    return _config.load_obj("scram", name, obj)

def load(*ref: str) -> scrambler.Scram:
    """Load a scrambler by name, `load("dvb_energy")`; a leading category still works, `load("adt", "dvb_energy")`"""
    *folder, name = ref
    return _config.load("scram", name, folder[0] if folder else None)
