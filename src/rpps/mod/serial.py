"""Modulation de-serialization helpers"""
from pyboiler.config import config
import pathlib
import json

from . import modulation
from .constellation import Mapping, Maps, Points

def identify():
    """Identify available modulation types"""
    mods = {}
    for folder in (config().PATH_CONFIG / "mod").iterdir():
        mods[folder.name] = [file.stem for file in folder.iterdir()]
    return mods

def load(name: str):
    """Load a modulation from a file name"""
    def load_complex(comp):
        c = []
        for num in comp:
            c.append(num["real"] + num["imag"] * 1j)
        return c
    folder = name[-3:].lower()
    mod = json.loads((config().PATH_CONFIG / "mod" / folder / f"{name.lower()}.json").read_text())

    return getattr(modulation, folder.upper()).load(name, mod)
