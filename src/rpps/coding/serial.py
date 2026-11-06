"""Coding (FEC) de-serialization helpers

Definitions live in config/coding/<kind>/<family>/<name>.json
 kind: block | conv | interleave   (blk and cnv still work as aliases)
 family: rs, bch, ldpc, hamming, repeat | dvbs, iess308, cnv | forney
Decoding only runs from a definition loaded here; `analysis.fec.Guess.save` is how a guessed structure becomes one.
CRC and framing are in rpps.encap.
"""
from pathlib import Path
import json

from pyboiler.config import config

from . import coding
from . import types as types

KINDS = ("block", "conv", "interleave")
ALIAS = {"blk": "block", "cnv": "conv"} # Short names still work
_LOADERS = {"block": coding.Block, "conv": coding.Convolutional, "interleave": coding.Interleaver}

def root() -> Path:
    """Directory holding the coding definitions"""
    return config().PATH_CONFIG / "coding"  # type: ignore

def ls(path=None):
    """List available FEC definitions by family: {"block/rs": [names], "conv/dvbs": [names], ...}"""
    base = Path(path) if path else root()
    codings = {}
    for kind in sorted(base.iterdir()):
        if not kind.is_dir():
            continue
        direct = sorted(f.stem for f in kind.iterdir() if f.suffix == ".json")
        if direct:
            codings[kind.name] = direct
        for sub in sorted(kind.iterdir()):
            if sub.is_dir():
                codings[f"{kind.name}/{sub.name}"] = sorted(f.stem for f in sub.iterdir() if f.suffix == ".json")
    return codings

def names(kind: str, sub=None, path=None):
    """Names of the definitions of a kind ("block", "conv", "interleave"), across families, or just one family"""
    kind = ALIAS.get(kind, kind)
    out = []
    for key, v in ls(path).items():
        k, _, s = key.partition("/")
        if k == kind and (sub is None or s == sub):
            out += v
    return sorted(out)

def init(name: str, obj: dict, decoder=None) -> coding.Coding:
    """Initialize coding. decoder: a decoder type to use instead of the definition's (see `coding.Block.load`)"""
    base = ALIAS.get(obj["base"], obj["base"])
    if base not in _LOADERS:
        raise NotImplementedError(f"Unknown coding: {name}, {obj}")
    return _LOADERS[base].load(name, obj, decoder)

def find(folder: str, name: str, path=None) -> Path:
    """File of a definition: config/coding/<kind>/[<family>/]<name>.json"""
    base = (Path(path) if path else root()) / ALIAS.get(folder, folder)
    fname = f"{name.lower()}.json"
    if (base / fname).exists():
        return base / fname
    for f in sorted(base.rglob(fname)):
        return f
    raise FileNotFoundError(f"{folder}/{name}: no {fname} under {base}")

def load(folder: str, name: str, path=None, decoder=None) -> coding.Coding:
    """Load a coding from config/coding/<kind>/**/<name>.json (or from `path`/<kind>/ instead of config/coding)

    decoder: decode with another type than the definition's: "bcjr" or "viterbi" for conv, "trellis" for linear block codes
    """
    code = json.loads(find(folder, name, path).read_text())
    if "." in name:
        name = name.split(".")[0]
    return init(name, code, decoder)

def generate(t_str):
    """Generate coding matrixes"""
    return getattr(types, f"generate_{t_str}")
