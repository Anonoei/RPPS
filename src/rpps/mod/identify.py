from . import MAP, MAPPING


def mapping(name: str, idx: int):
    return MAPPING[name[-3:]][name][idx]


def by_name(name: str, idx = None):
    mod = MAP[name[-3:]][name]
    if idx is not None:
        mod = mod(mapping(name, idx))
    else:
        mod = mod()
    return mod
