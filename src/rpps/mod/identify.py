from . import MAP

def by_name(name: str):
    return MAP[name[-3:]][name]
