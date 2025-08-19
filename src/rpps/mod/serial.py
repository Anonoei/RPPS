"""Modulation de-serialization helpers"""
from pyboiler.config import config
import json
import typing

from . import types
from .modulation import Modulation
from .constellation import ModConstellation

def identify():
    """Identify available modulation types"""
    mods = {}
    for folder in (config().PATH_CONFIG / "mod").iterdir(): # type: ignore
        mods[folder.name] = [file.stem for file in folder.iterdir()]
    return mods


def init(name, obj: dict) -> Modulation:
    """Initialize modulation"""
    return getattr(types, obj["base"]).load(name, obj)


def load(name: str) -> ModConstellation:
    """Load a modulation from a file name"""
    # returning ModConstellation hides errors on client side
    folder = name[-3:].lower()

    mod = json.loads((config().PATH_CONFIG / "mod" / folder / f"{name.lower()}.json").read_text())  # type: ignore

    return init(name, mod) # type: ignore
