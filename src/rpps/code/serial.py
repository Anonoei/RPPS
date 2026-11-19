"""Code definitions, two resources: config/fec/<kind>/<family>/<name>.json (`rp.code.load("block/rs/204_188")`) and
config/interleave/<type>/<name>.json (`rp.code.load("interleave/forney/12_17")`)

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
                  table), "shifts" + "z" (quasi-cyclic: a matrix of shifts of the z x z identity, -1 a zero block: 802.11n, 802.16e)
                  or "gallager": {...}; optional "k" + "systematic": "head" | "tail" | [positions], "iters": 50, "alpha": 0.8
    conv        {"base": "conv", "constraint": K, "generator": den rows of K bits, "puncture": den rows, "decoder": "viterbi"}
                  one set of keys for the encoder and its decoder ("viterbi", or "bcjr": soft output; also "maxlog",
                  "hard_llr", "termination", "invert"). A mother code is a file (fec/conv/k7_171_133.json); its punctured
                  rates extend it with their "puncture" (fec/conv/k7_171_133/3_4.json)
    interleave  {"base": "interleave", "type": "forney", "I": 12, "M": 17, "symbol": 8}
                {"base": "interleave", "type": "block", "perm": [...] or {"each": "<i>", "n": N}, "map": "from" | "to", "symbol": 1}
                {"base": "interleave", "type": "block", "length": D, "cols": 32, "col_perm": [...], "fill": "front"}
    Any value can be a load-time parameter: "params": {"ncbps": null}, then {"param": "ncbps"} or {"expr": "..."}.

A reference is the path below config/fec or config/interleave with the kind first: "block/rs/204_188", "conv/k7_171_133/3_4",
"interleave/forney/12_17". A bare name ("204_188") works while one definition of any kind has it.

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
FEC = _config.register_domain("fec", {"block": coding.Block.load, "conv": coding.Convolutional.load})
INTERLEAVE = _config.register_domain("interleave", {"interleave": coding.Interleaver.load})

def _domain(kind):
    return "interleave" if kind == "interleave" else "fec"

def root() -> Path:
    """Directory holding the coding definitions"""
    return _config.find_config("fec")

def ls(path=None):
    """List available FEC definitions by family: {"block/rs": [names], "conv/dvbs": [names], ...}"""
    return {**_config.ls("fec", path), **{f"interleave/{k}" if k != "interleave" else k: v
                                              for k, v in _config.ls("interleave", path).items()}}

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

def _find(kind, rest, path):
    dom = _domain(kind)
    file, parts = _config.locate(dom, rest, None if dom == "interleave" else kind, path)
    if file is None:
        return None, None
    return file, "/".join(parts if dom == "interleave" else parts[1:])

def locate(ref: str, path=None):
    """(file, kind, name) of a definition: `ref` is "<kind>/<path below it>" ("block/rs/204_188", "interleave/forney/12_17"), or
    a bare name one definition of any kind has. name is the path below the kind ("rs/204_188"): what the loaded code is called"""
    ref = str(ref).strip("/")
    kind, _, rest = ref.partition("/")
    if kind in KINDS and rest:
        file, name = _find(kind, rest, path)
        hits = [(file, kind, name)] if file is not None else []
    else:
        hits = [(f, k, n) for k in KINDS for f, n in [_find(k, ref, path)] if f is not None]
    if len(hits) > 1:
        raise ConfigError(f"{ref} is ambiguous: {[f'{k}/{n}' for _, k, n in hits]}")
    if not hits:
        raise FileNotFoundError(f"{ref}: no code definition (give <kind>/<name>, kind one of {KINDS}), see rp.code.ls()")
    return hits[0]

def find(ref: str, path=None) -> Path:
    """File of a definition (see `locate`)"""
    return locate(ref, path)[0]

def load(ref: str, path=None, decoder=None, **params) -> coding.Coding:
    """Load a code: `load("block/rs/204_188")`, `load("conv/k7_171_133/3_4")`, `load("interleave/forney/12_17")`
    (config/fec/<kind>/..., config/interleave/...; or below `path` instead of the shipped config)

    decoder: decode with another type than the definition's: "bcjr" or "viterbi" for conv, "trellis" for linear block codes
    params: values of the definition's "params" (`load("interleave/block/wifi_ofdm", ncbps=96, nbpsc=2)`)
    """
    file, kind, name = locate(ref, path)
    return init(name, _config.resolve(_config.read(file), params, f"{kind}/{name}"), decoder)
