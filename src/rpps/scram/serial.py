"""Scrambler definitions: config/scram/<category>/<name>.json, loaded with `rp.scram.load("dvb_energy")`

    fdt  feedthrough (self-synchronizing): the output feeds the register
         {"base": "fdt", "type": "fibonacci" | "galois_left" | "galois_right",
          "poly": [16, 14, 13, 11],        feedback taps, 1 = first stage
          "seed": [1, 0, 1, ...]}          register bits, first stage first
    adt  additive (synchronous): data xor a PRBS that ignores the data
         {"base": "adt", "poly": [15, 14], "seed": [1, 0, 0, ...] | "100101010000000",
          "frame": {"bytes": 1504,         optional: restart the PRBS every `bytes` bytes
                    "sync_period": 188, "sync_bytes": 1,   leading bytes of each packet the PRBS skips over
                    "skip_first": 1, "marker": 184,        skip_first: bytes before the PRBS starts (default sync_bytes);
                    "sync": 71}}                           marker: first byte of a frame as sent (DVB: the inverted sync, 0xB8);
                                                           sync: what it becomes when descrambled (default marker ^ 0xFF)
"""
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
