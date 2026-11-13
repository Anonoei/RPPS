"""Constellations: config/constellation/[<group>/]<name>.json, loaded with `rp.symbol.load("QPSK")`; physical headers:
config/header/physical/<name>.json, `rp.symbol.load_header("dvbs2_pl")`

A name with no file ("16QAM", "64QAM", "16APSK", "4ASK", "OOK") is generated instead (see `generate`).

    constellation   {"base": "constellation",
                     "points": [{"mag": 1, "pha": 45}, {"re": 0.7, "im": -0.7}, [-0.7, 0.7], ...],   where the points are: phase in degrees
                     "maps": [{"map": [0, 1, 3, 2], "comment": "R000 Grey"}, ...],                      the ways of labelling them with bits
                     "normalize": true}                                                                  scale to unit mean power (default)
        maps[i].map[point] is the value (bits, first bit most significant) the point carries. The first map is the default;
        `rp.symbol.load(name, "R090")` picks another by index or a word of its comment, and every rotation and spectral inversion of
        the first map is added (see rpps.symbol.mapping).
    header          {"base": "header", "type": "physical" | "link"}: known symbols then a short code carrying fields
                    (rpps.symbol.frame.Header), or a link header (rpps.link.header.Header): one resource, `header`
"""
import re

from .. import _config
from ..errors import ConfigError

from .modulation import Modulation
from .constellation import Constellation, PSK, QAM, APSK, ASK

from .frame import Header

DOMAIN = _config.register_domain("constellation", {"constellation": Constellation.load})

def _header(name, obj):
    t = obj.get("type")
    if t == "physical":
        return Header.load(name, obj)
    if t == "link":
        from ..link.header import Header as LinkHeader
        return LinkHeader.load(name, obj)
    raise ConfigError(f"header {name}: type is physical or link, not {t!r}")

HEADERS = _config.register_domain("header", {"header": _header})

def load_header(name: str, **params) -> Header:
    """A physical header definition: `load_header("dvbs2_pl")` (config/header/physical)"""
    h = _config.load("header", name, "physical", params=params)
    return h

def ls():
    """List available modulation types: {category: [names]}"""
    return _config.ls("constellation")


def init(name, obj: dict) -> Modulation:
    """Initialize modulation"""
    return _config.load_obj("constellation", name, obj)


_GENERATED = re.compile(r"^(?:(\d+)\s*[-_]?\s*)?(B?PSK|QPSK|QAM|APSK|ASK|OOK)(?:[-_]?(\d+))?$", re.I)

def generate(name: str) -> Constellation:
    """Make a constellation from its name: "16QAM", "QAM64", "8PSK", "16APSK", "4ASK", "OOK"... (no definition file)"""
    m = _GENERATED.match(name.strip())
    if m is None:
        raise ConfigError(f"No modulation {name!r}: not a definition ({ls()}) and not a name like 16QAM, 8PSK, 16APSK, 4ASK")
    order, kind = m.group(1) or m.group(3), m.group(2).upper()
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
    """Load a modulation by name: a definition in config/constellation (QPSK, 8PSK, DVBS2_8PSK...) or a generated one (16QAM, 64QAM, 16APSK, 4ASK)

    mapping: which bit map, if not the first: an index, or a word of its comment (`load("QPSK", "R090")`)"""
    mod = _load(name)
    if mapping is not None:
        mod.set_mapping(mapping)
    return mod

def _load(name: str) -> Constellation:
    path = _config.find("constellation", name)
    if path is None:
        return generate(name)
    return init(name, _config.read(path)) # type: ignore
