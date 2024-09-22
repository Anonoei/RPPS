"""Modulation de-serialization helpers"""
from pyboiler.config import config
import pathlib
import json

from . import coding
from . import _types as types

def identify():
    """Identify available coding types"""
    codings = {}
    for folder in (config().PATH_CONFIG / "coding").iterdir():  # type: ignore
        codings[folder.name] = [file.stem for file in folder.iterdir()]
    return codings


def init(name: str, obj: dict):
    if obj["base"] == "blk":
        return coding.Block.load(name, obj)
    return coding.Coding(1,1)

def load(folder: str, name: str) -> coding.Coding:
    """Load a coding from a file name"""
    code = json.loads((config().PATH_CONFIG / "coding" / folder / f"{name.lower()}.json").read_text())  # type: ignore

    return init(name, code)


def generate(type):
    return getattr(types, f"generate_{type}")
