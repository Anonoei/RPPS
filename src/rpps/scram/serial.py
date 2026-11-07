"""Modulation de-serialization helpers"""
import json

from .. import _config
from ..errors import ConfigError

from . import scrambler

def ls():
    """List available scram types"""
    scrams = {}
    for layer in _config.layers("scram"):
        for folder in layer.iterdir():
            scrams[folder.name] = sorted(set(scrams.get(folder.name, [])) | {file.stem for file in folder.iterdir()})
    return scrams


def init(name: str, obj: dict) -> scrambler.Scram:
    """Initialize scram"""
    if obj["base"] == "fdt":
        return scrambler.Feedthrough.load(name, obj)
    elif obj["base"] == "adt":
        return scrambler.Additive.load(name, obj)
    raise ConfigError(f"Unknown scrambler base {obj.get('base')!r} in {name}: have fdt, adt")

def load(folder: str, name: str) -> scrambler.Scram:
    """Load a scrambler from a file name"""
    path = _config.find_file("scram", folder, f"{name.lower()}.json")
    if path is None:
        raise ConfigError(f"No scrambler {folder}/{name}, have {ls()}")
    scram = json.loads(path.read_text())

    return init(name, scram)
