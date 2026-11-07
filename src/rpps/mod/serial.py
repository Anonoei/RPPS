"""Modulation definitions: config/mod/<category>/<name>.json, loaded with `rp.mod.load("QPSK")`

A name with no file ("16QAM", "64QAM", "16APSK", "4ASK", "OOK") is generated instead (see `generate`).

    psk   {"base": "psk", "Points": [{"mag": 1, "pha": 45}, ...],         constellation points, phase in degrees
           "Maps": [{"map": [0, 1, 3, 2], "comment": "R000 Grey"}, ...]}  the ways of labelling the points with bits
          The first map is the default; `rp.mod.load(name, "R090")` picks another by index or a word of its comment.
"""
import re

from .. import _config
from ..errors import ConfigError
import typing

from . import types
from .modulation import Modulation
from .constellation import ModConstellation

DOMAIN = _config.register_domain("mod", {"psk": types.PSK.load})

def ls():
    """List available modulation types: {category: [names]}"""
    return _config.ls("mod")


def init(name, obj: dict) -> Modulation:
    """Initialize modulation"""
    return _config.load_obj("mod", name, obj)


_GENERATED = re.compile(r"^(?:(\d+)\s*[-_]?\s*)?(B?PSK|QPSK|QAM|APSK|ASK|OOK)(?:[-_]?(\d+))?$", re.I)

def generate(name: str) -> ModConstellation:
    """Make a constellation from its name: "16QAM", "QAM64", "8PSK", "16APSK", "4ASK", "OOK"... (no definition file)"""
    m = _GENERATED.match(name.strip())
    if m is None:
        raise ConfigError(f"No modulation {name!r}: not a definition ({ls()}) and not a name like 16QAM, 8PSK, 16APSK, 4ASK")
    order, kind, tail = m.group(1) or m.group(3), m.group(2).upper(), None
    M = int(order) if order else None
    if kind == "BPSK":
        return types.PSK(2, offset=0.0)
    if kind == "QPSK":
        return types.PSK(4)
    if kind == "OOK":
        return types.ASK(2)
    if M is None:
        raise ConfigError(f"{name!r} needs an order, like 16{kind}")
    return {"PSK": types.PSK, "QAM": types.QAM, "APSK": types.APSK, "ASK": types.ASK}[kind](M)

def load(name: str, mapping=None) -> ModConstellation:
    """Load a modulation by name: a definition in config/mod (QPSK, 8PSK, DVBS2_8PSK...) or a generated one (16QAM, 64QAM, 16APSK, 4ASK)

    mapping: which bit map, if not the first: an index, or a word of its comment (`load("QPSK", "R090")`)"""
    # returning ModConstellation hides errors on client side
    mod = _load(name)
    if mapping is not None:
        mod.set_mapping(mapping)
    return mod

def _load(name: str) -> ModConstellation:
    path = _config.find("mod", name)
    if path is None:
        return generate(name)
    return init(name, _config.read(path)) # type: ignore
