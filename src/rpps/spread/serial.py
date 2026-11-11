"""Spreading code definitions: config/spread/<category>/<name>.json, loaded with `rp.spread.load("barker11")`

    fixed   a sequence written out ("chips" +/-1, or "bits" with "one")
    pnset   a table of sequences, one per symbol value
    gold    register sequences XOR'd, with an optional parameter that picks the code of a family (GPS "prn")
See rpps.spread.code for the keys. Chip rates are in "rate" (chips/s).
"""
from .. import _config

from . import code

DOMAIN = _config.register_domain("spread", {"fixed": code.load_fixed, "pnset": code.load_pnset, "gold": code.load_gold})

def ls():
    """List available spreading codes: {category: [names]}"""
    return _config.ls("spread")

def init(name: str, obj: dict) -> code.Code:
    """Initialize a code from a definition object"""
    return _config.load_obj("spread", name, obj)

def load(*ref: str, **params) -> code.Code:
    """Load a spreading code by name, `load("barker11")`; params: values of the definition's "params" (`load("gps_ca", prn=1)`)"""
    *folder, name = ref
    return _config.load("spread", name, folder[0] if folder else None, params=params)
