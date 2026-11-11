"""Coding (FEC) definitions: config/coding/<kind>/<family>/<name>.json, loaded with `rp.code.load("block", "rs/204_188")`

Folders are kind (`base`), then the structure of the code; the standard goes in `ref`/`comment`. Names are parameters.

    block       {"base": "block", "type": ..., "decoder": optional type to decode with instead}
      linear      "code": [n, k], "p": parity block rows    or    "g": generator rows, "h": check rows (decoded on its syndrome "trellis" when small)
      cyclic      "n", "k", "poly": generator polynomial (bit i = x^i), "extend": overall parity bit, "shorten": leading
                  message bits dropped, "t": errors to correct (default from d_min); decoded by "syndrome" table
      repeat      "count": 3
      fsm         trellis code as a state machine: "k", "n", "next", "output" tables, "start", "end", "tail",
                  "symbols": {"bits", "levels"} to decode amplitudes (P25 trellis, ATSC 8VSB)
      turbo       LTE turbo code: "length", "g0", "g1", "constraint", "qpp" or "qpp_table", "iters", "scale"
      ratematch   LTE rate matching: "mode": "conv" | "turbo", "length" (bits per stream), "out" (E), "rv", "ncb"
      product     block product code: "rows", "cols" (block definitions), "reserved", "lead", "iters" (DMR BPTC)
      rs          "n", "k", "m": 8, "poly": field polynomial, "fcr": 0, "prim": 1, "puncture": codeword positions not
                  sent (negative from the end; decoded as erasures)
    Every block decoder leaves `errors` per codeword after a decode: bits (RS: symbols) corrected, -1 for a failure.
      bch         "n", "k", "t", "m": 16, "poly": "0x1002D"
      ldpc        "n" and the checks as "checks" (positions per check), "h" (dense rows), "table" + "k" + "group" (DVB-S2 address
                  table) or "gallager": {...}; optional "k" + "systematic": "head" | "tail" | [positions], "iters": 50, "alpha": 0.8
    conv        {"base": "conv", "type": "split", "encode": {...}, "decode": {...}} sharing "puncture" and "terminated"
      encode      "type": "conv", "num", "den", "constraint", "generator": den rows of K bits, "puncture": den rows
      decode      "type": "viterbi" | "bcjr" (soft output; also "maxlog", "hard_llr"), same keys as encode
    interleave  {"base": "interleave", "type": "forney", "I": 12, "M": 17, "symbol": 8}
                {"base": "interleave", "type": "block", "perm": [...] or {"each": "<i>", "n": N}, "map": "from" | "to", "symbol": 1}
                {"base": "interleave", "type": "block", "length": D, "cols": 32, "col_perm": [...], "fill": "front"}
    Any value can be a load-time parameter: "params": {"ncbps": null}, then {"param": "ncbps"} or {"expr": "..."}.

Decoding only runs from a definition loaded here; `analysis.fec.Guess.save` turns a guessed structure into one,
`rp.code.import_alist` an LDPC alist file. CRC and framing are in rpps.link.
"""
from pathlib import Path


from .. import _config
from ..errors import ConfigError
from . import coding
from . import types as types

KINDS = ("block", "conv", "interleave")
_LOADERS = {"block": coding.Block, "conv": coding.Convolutional, "interleave": coding.Interleaver}
DOMAIN = _config.register_domain("coding", {k: v.load for k, v in _LOADERS.items()})

def root() -> Path:
    """Directory holding the coding definitions"""
    return _config.find_config("coding")

def ls(path=None):
    """List available FEC definitions by family: {"block/rs": [names], "conv/dvbs": [names], ...}"""
    return _config.ls("coding", path)

def names(kind: str, sub=None, path=None):
    """References of the definitions of a kind ("block", "conv", "interleave"), as `load(kind, ref)` takes them ("rs/204_188"), across families or just one"""
    out = []
    for key, v in ls(path).items():
        k, _, family = key.partition("/")
        if k == kind and (sub is None or family == sub):
            out += [f"{family}/{n}" if family else n for n in v]
    return sorted(out)

def init(name: str, obj: dict, decoder=None) -> coding.Coding:
    """Initialize coding. decoder: a decoder type to use instead of the definition's (see `coding.Block.load`)"""
    base = str(obj["base"]).lower()
    if base not in _LOADERS:
        raise ConfigError(f"Unknown coding base {obj.get('base')!r} in {name}: have {sorted(_LOADERS)}")
    return _LOADERS[base].load(name, obj, decoder)

def locate(folder: str, ref: str, path=None):
    """(file, name) of a definition: config/coding/<kind>/[<family>/]<name>.json. The name of what is loaded is its family folder, else its file"""
    file, parts = _config.locate("coding", ref, folder, path)
    if file is None:
        raise FileNotFoundError(f"{folder}/{ref}: not found under {folder} in {[str(r) for r in ([Path(path)] if path else _config.layers('coding'))]}")
    family = parts[1:-1]
    return file, family[-1] if family else parts[-1]

def find(folder: str, ref: str, path=None) -> Path:
    """File of a definition: config/coding/<kind>/[<family>/]<name>.json"""
    return locate(folder, ref, path)[0]

def load(folder: str, ref: str, path=None, decoder=None, **params) -> coding.Coding:
    """Load a coding from config/coding/<kind>/<family>/<name>.json, `load("block", "rs/204_188")` (or from `path`/<kind>/ instead of config/coding)

    A bare name ("204_188") also works when only one family has it.
    decoder: decode with another type than the definition's: "bcjr" or "viterbi" for conv, "trellis" for linear block codes
    params: values of the definition's "params" (`load("interleave", "wifi_ofdm", ncbps=96, nbpsc=2)`)
    """
    file, name = locate(folder, ref, path)
    return init(name, _config.resolve(_config.read(file), params, f"coding/{folder}/{ref}"), decoder)

def generate(t_str):
    """Generate coding matrixes"""
    return getattr(types, f"generate_{t_str}")
