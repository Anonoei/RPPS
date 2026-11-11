"""The sequence resource (ARCHITECTURE.md, Config resources): bit and chip sequences, config/sequence/<type>/<name>.json

One engine for every operator that uses a known sequence: a scrambler XORs one in (`rp.code.scram`), a spreader multiplies
by one (`rp.symbol.Spread`), a phase scrambler turns symbols by one (`rp.symbol.PhaseScramble`), a code-phase search looks
for one (`rp.sync.acquire_code`).

    rp.sequence.load("lte_gold", c_init=77).bits(100)       a Recurrence: 0/1 bits from any position
    rp.sequence.load("gps_ca", prn=1).chips                 a Code: (count, length) of +/-1

    {"base": "sequence", "type": "recurrence", "sequences": [...], "skip": 1600}      see rpps.sequence.recurrence
    {"base": "sequence", "type": "chips" | "set" | "gold", ...}                       see rpps.sequence.code
Any value can be a load-time parameter ("params", then {"param": ...} / {"expr": ...}). Chip rates are in "rate" (chips/s).
"""
from .. import _config
from ..errors import ConfigError
from . import lfsr
from .recurrence import Recurrence
from .code import Code, LOADERS as _CODES

def _load(name, obj):
    t = obj.get("type")
    if t == "recurrence":
        return Recurrence.load(name, obj)
    if t in _CODES:
        return _CODES[t](name, obj)
    raise ConfigError(f"sequence {name}: no type {t!r} (have recurrence, {', '.join(_CODES)})")

DOMAIN = _config.register_domain("sequence", {"sequence": _load})

def ls():
    """The sequences there are: {type folder: [names]}"""
    return _config.ls("sequence")

def init(name, obj):
    """A sequence from a definition object"""
    return _config.load_obj("sequence", name, obj)

def load(name, **params):
    """Load a sequence by name: `load("barker11")`, `load("gps_ca", prn=1)`. params: values of its "params" """
    return _config.load("sequence", name, params=params)
