"""Constellations: config/constellation/[<group>/]<name>.json, loaded with `rp.symbol.load("QPSK")`; physical headers:
config/header/physical/<name>.json, `rp.symbol.load_header("dvbs2_pl")`

`load` only loads definitions; the usual shapes by name ("16QAM", "64QAM", "16APSK", "4ASK", "OOK") come from `generate`, so a
DVB-S2 16APSK (`load("apsk/dvbs2_16apsk_2_3")`) is never confused with the generic one (`generate("16APSK")`).

    constellation   {"base": "constellation",
                     "points": [{"mag": 1, "pha": 45}, {"re": 0.7, "im": -0.7}, [-0.7, 0.7], ...],   where the points are: phase in degrees
                     or "rings": [{"n": 4, "mag": 1, "pha": 45}, {"n": 12, "mag": 2.85, "pha": 15}],    PSK / APSK rings, innermost first
                     "rotate": -45,                                                                      degrees added to every point
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
from ..link.serial import HEADERS # The header resource (physical and link), registered with the link definitions

DOMAIN = _config.register_domain("constellation", {"constellation": Constellation.load})

def load_header(name: str, **params) -> Header:
    """A physical header definition: `load_header("dvbs2_pl")` (config/header/physical)"""
    h = _config.load("header", name, "physical", params=params)
    return h

def ls():
    """List available modulation types: {category: [names]}"""
    return _config.ls("constellation")


def load_or_generate(name: str, mapping=None) -> Constellation:
    """A definition named `name`, else the shape `generate` makes of it: for names that come out of a detector ("8PSK",
    "16QAM"), where either can turn up. Code that knows which it wants calls `load` or `generate`"""
    if _config.find("constellation", name) is not None:
        return load(name, mapping)
    mod = generate(name)
    if mapping is not None:
        mod.set_mapping(mapping)
    return mod

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
        return PSK(2)
    if kind == "QPSK":
        return PSK(4)
    if kind == "OOK":
        return ASK(2)
    if M is None:
        raise ConfigError(f"{name!r} needs an order, like 16{kind}")
    return {"PSK": PSK, "QAM": QAM, "APSK": APSK, "ASK": ASK}[kind](M)

def load(ref: str, mapping=None, **params) -> Constellation:
    """Load a constellation definition, config/constellation/<group>/<name>.json: `load("psk/qpsk")`, `load("apsk/dvbs2_16apsk", gamma=2.85)`
    (a bare name works while unique). For the usual shapes by name use `generate("16QAM")`.

    mapping: which bit map, if not the first: an index, or a word of its comment (`load("psk/qpsk", "R090")`)
    params: values of the definition's "params" (see rp._config.resolve)"""
    path = _config.find("constellation", ref)
    if path is None:
        hint = f"; {ref!r} is a shape: rp.symbol.generate({ref!r})" if _GENERATED.match(str(ref).strip()) else ""
        raise ConfigError(f"constellation/{ref}: no definition{hint}; have {ls()}")
    mod = init(ref, _config.resolve(_config.read(path), params, f"constellation/{ref}"))
    if mapping is not None:
        mod.set_mapping(mapping)
    return mod # type: ignore
