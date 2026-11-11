"""Modulation definitions: config/mod/<category>/[<group>/]<name>.json, loaded with `rp.symbol.load("QPSK")`

A name with no file ("16QAM", "64QAM", "16APSK", "4ASK", "OOK") is generated instead (see `generate`).

    constellation   {"base": "constellation",
                     "points": [{"mag": 1, "pha": 45}, {"re": 0.7, "im": -0.7}, [-0.7, 0.7], ...],   where the points are: phase in degrees
                     "maps": [{"map": [0, 1, 3, 2], "comment": "R000 Grey"}, ...],                      the ways of labelling them with bits
                     "normalize": true}                                                                  scale to unit mean power (default)
        maps[i].map[point] is the value (bits, first bit most significant) the point carries. The first map is the default;
        `rp.symbol.load(name, "R090")` picks another by index or a word of its comment, and every rotation and spectral inversion of
        the first map is added (see rpps.symbol.mapping).
    header          a physical header: known symbols then a short code carrying fields (see rpps.symbol.frame.Header)
"""
import re

from .. import _config
from ..errors import ConfigError
import typing

from .modulation import Modulation
from .constellation import Constellation, PSK, QAM, APSK, ASK

from .frame import Header

DOMAIN = _config.register_domain("mod", {"constellation": Constellation.load, "header": Header.load})

def ls():
    """List available modulation types: {category: [names]}"""
    return _config.ls("mod")


def init(name, obj: dict) -> Modulation:
    """Initialize modulation"""
    return _config.load_obj("mod", name, obj)


_GENERATED = re.compile(r"^(?:(\d+)\s*[-_]?\s*)?(B?PSK|QPSK|QAM|APSK|ASK|OOK)(?:[-_]?(\d+))?$", re.I)

def generate(name: str) -> Constellation:
    """Make a constellation from its name: "16QAM", "QAM64", "8PSK", "16APSK", "4ASK", "OOK"... (no definition file)"""
    m = _GENERATED.match(name.strip())
    if m is None:
        raise ConfigError(f"No modulation {name!r}: not a definition ({ls()}) and not a name like 16QAM, 8PSK, 16APSK, 4ASK")
    order, kind, tail = m.group(1) or m.group(3), m.group(2).upper(), None
    M = int(order) if order else None
    if kind == "BPSK":
        return PSK(2, offset=0.0)
    if kind == "QPSK":
        return PSK(4)
    if kind == "OOK":
        return ASK(2)
    if M is None:
        raise ConfigError(f"{name!r} needs an order, like 16{kind}")
    return {"PSK": PSK, "QAM": QAM, "APSK": APSK, "ASK": ASK}[kind](M)

def load(name: str, mapping=None) -> Constellation:
    """Load a modulation by name: a definition in config/mod (QPSK, 8PSK, DVBS2_8PSK...) or a generated one (16QAM, 64QAM, 16APSK, 4ASK)

    mapping: which bit map, if not the first: an index, or a word of its comment (`load("QPSK", "R090")`)"""
    mod = _load(name)
    if mapping is not None:
        mod.set_mapping(mapping)
    return mod

def _load(name: str) -> Constellation:
    path = _config.find("mod", name)
    if path is None:
        return generate(name)
    return init(name, _config.read(path)) # type: ignore
