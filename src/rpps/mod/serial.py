"""Modulation de-serialization helpers"""
from pyboiler.config import config
import json

from . import modulation

def identify():
    """Identify available modulation types"""
    mods = {}
    for folder in (config().PATH_CONFIG / "mod").iterdir(): # type: ignore
        mods[folder.name] = [file.stem for file in folder.iterdir()]
    return mods


def init(name, obj: dict) -> modulation.Modulation:
    """Initialize modulation"""
    return getattr(modulation, obj["base"]).load(name, obj)


def load(name: str) -> modulation.Modulation:
    """Load a modulation from a file name"""
    folder = name[-3:].lower()

    mod = json.loads((config().PATH_CONFIG / "mod" / folder / f"{name.lower()}.json").read_text())  # type: ignore

    return init(name, mod)
