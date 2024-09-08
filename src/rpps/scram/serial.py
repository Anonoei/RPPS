"""Modulation de-serialization helpers"""
from pyboiler.config import config
import pathlib
import json

from . import scrambler

def identify():
    """Identify available scram types"""
    mods = {}
    for folder in (config().PATH_CONFIG / "scram").iterdir():  # type: ignore
        mods[folder.name] = [file.stem for file in folder.iterdir()]
    return mods

def load(folder: str, name: str) -> scrambler.Scram:
    """Load a modulation from a file name"""
    def load_complex(comp):
        c = []
        for num in comp:
            c.append(num["real"] + num["imag"] * 1j)
        return c
    scram = json.loads((config().PATH_CONFIG / "scram" / folder / f"{name.lower()}.json").read_text())  # type: ignore

    if folder == "fdt":
        return scrambler.Feedthrough.load(name, scram)
    elif folder == "adt":
        return scrambler.Additive.load(name, scram)
    return scrambler.Scram()
