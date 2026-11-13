"""Scrambler definitions: config/scramble/<type>/<name>.json, loaded with `rp.code.scram.load("dvb_energy")`

How a sequence is applied to bits; the sequences are their own resource (rpps.sequence), which a scrambler names.

    additive     data XOR a sequence that ignores the data
         {"base": "scramble", "type": "additive",
          "sequence": "dvb_energy" | {"name": "lte_gold", "args": {"c_init": {"expr": "..."}}},   with its parameters
          "period": 1920,                    optional: restart every `period` bits
          "frame": {"bytes": 1504, "sync_period": 188, "sync_bytes": 1, "marker": 184}}  optional: packet structure (see Additive)
    feedthrough  self-synchronizing: the output feeds the register, so the register is the scrambler's own
         {"base": "scramble", "type": "feedthrough", "register": "fibonacci" | "galois_left" | "galois_right",
          "poly": [16, 14, 13, 11],          feedback taps, 1 = first stage
          "seed": [1, 0, 1, ...]}            register bits, first stage first
    Seeds, sequence arguments and any other value can be load-time parameters ("params", then {"param": ...} / {"expr": ...}).
"""
from ... import _config
from ...errors import ConfigError

from . import scrambler

def _load(name, obj):
    t = obj.get("type")
    if t == "additive":
        return scrambler.Additive.load(name, obj)
    if t == "feedthrough":
        return scrambler.Feedthrough.load(name, obj)
    raise ConfigError(f"scramble {name}: no type {t!r} (have additive, feedthrough)")

DOMAIN = _config.register_domain("scramble", {"scramble": _load})

def ls():
    """The scramblers there are: {type: [names]}"""
    return _config.ls("scramble")

def init(name: str, obj: dict) -> scrambler.Scram:
    """A scrambler from a definition object"""
    return _config.load_obj("scramble", name, obj)

def load(name: str, **params) -> scrambler.Scram:
    """Load a scrambler by name: `load("dvb_energy")`, `load("ble_whiten", channel=37)`. params: values of its "params" """
    return _config.load("scramble", name, params=params)
