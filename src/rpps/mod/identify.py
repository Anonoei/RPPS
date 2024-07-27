from . import MAP, MAPPING

from .constellation import Mapping
from .modulation import Modulation


def mapping(name: str, idx: int) -> Mapping:
    return MAPPING[name[-3:]][name][idx]


def by_name(name: str, idx: int = -123) -> Modulation:
    mod = MAP[name[-3:]][name]
    if idx is not -123:
        mod = mod(mapping(name, idx))
    else:
        mod = mod()
    return mod
