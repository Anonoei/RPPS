"""Modulation de-serialization helpers"""
from pyboiler.config import config
import json

from . import scrambler

def identify():
    """Identify available scram types"""
    scrams = {}
    for folder in (config().PATH_CONFIG / "scram").iterdir():  # type: ignore
        scrams[folder.name] = [file.stem for file in folder.iterdir()]
    return scrams


def init(name: str, obj: dict) -> scrambler.Scram:
    """Initialize scram"""
    if obj["base"] == "fdt":
        return scrambler.Feedthrough.load(name, obj)
    elif obj["base"] == "adt":
        return scrambler.Additive.load(name, obj)
    return scrambler.Scram()


def load(folder: str, name: str) -> scrambler.Scram:
    """Load a scrambler from a file name"""
    scram = json.loads((config().PATH_CONFIG / "scram" / folder / f"{name.lower()}.json").read_text())  # type: ignore

    return init(name, scram)
