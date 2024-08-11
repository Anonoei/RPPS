from . import MAP, MAPPING

from .constellation import Mapping, Maps
from .modulation import Modulation

def maps(n: str) -> Maps:
    return MAPPING[n[-3:]][n]


def mapping(n: str, idx: int) -> Mapping:
    return maps(n)[idx]


def name(n: str, idx: int = -123) -> Modulation:
    mod = MAP[n[-3:]][n]
    if idx == -123:
        mod = mod()
    else:
        mod = mod(mapping(n, idx))
    return mod
