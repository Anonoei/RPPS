"""Link definitions, one resource each: config/crc, config/frame, config/header/link; `rp.link.load("crc/crc32_mpeg2")`

    crc    {"base": "crc", "type": "crc", "width": 16, "poly": "0x1021", "init": "0x0", "xorout": "0x0",
            "refin": false, "refout": false,       the reveng catalogue fields (numbers or hex strings)
            "check": "0x29b1"}                      optional: the CRC of b"123456789", verified on load
           width is 1 to 64 bits; encode / decode / syndrome work on bit streams of any length (see `crc`)
           {"base": "crc", "type": "composite",     a check from CRC parts over transformed views of the message
            "parts": [{"crc": "crc8_smbus"}, {"crc": "crc8_smbus", "input": ["nibble_swap"]}],
            "combine": "concat"}                    (or "xor"); goes wherever a crc does (see `rpps.link.check`)
           {"base": "crc", "type": "checksum",      a sum, ones' complement (Internet), Fletcher, Adler or XOR checksum
            "algorithm": "internet", ...}           (see `rpps.link.checksum`); goes wherever a crc does
    frame  {"base": "frame", "type": "hdlc", "flag": "0x7E", "stuff": 5, "nrzi": false,
            "fcs": "crc16_x25",                     the name of a crc definition
            "min_bytes": 4}
           {"base": "frame", "type": "bytes",       byte stuffing (SLIP, KISS, asynchronous PPP; see `bytestuff`)
            "flag": "0xC0", "escape": "0xDB",
            "escapes": {"0xC0": "0xDC", "0xDB": "0xDD"},   what follows the escape for each byte, or "xor": "0x20" for
            "fcs": null, "refin": false}                   the bytes in "escaped"; an optional fcs; refin: LSB first
    header {"base": "header", "type": "link", "fields": [...], "crc": ..., "payload": "..."}   fields, CRC and payload length of a
            frame's header, driven and published through a profile (see rpps.link.header)
"""
from pathlib import Path

from .. import _config
from ..errors import ConfigError
from .encap import CRC, Frame

KINDS = ("crc", "frame", "header")

DOMAINS = {"crc": _config.register_domain("crc", {"crc": CRC.load}), "frame": _config.register_domain("frame", {"frame": Frame.load})}

def _header(name, obj):
    """A header definition: "physical" (known symbols and a short code: rp.symbol.Header) or "link" (fields, CRC, payload length)"""
    t = obj.get("type")
    if t == "physical":
        from ..symbol.frame import Header as PhysicalHeader
        return PhysicalHeader.load(name, obj)
    if t == "link":
        from .header import Header as LinkHeader
        return LinkHeader.load(name, obj)
    raise ConfigError(f"header {name}: type is physical or link, not {t!r}")

HEADERS = _config.register_domain("header", {"header": _header}) # One resource for both kinds, whichever package is imported first

def root() -> Path:
    """Directory holding the encapsulation definitions"""
    return _config.find_config("crc")

def ls(path=None, blind=False):
    """List available definitions: {category: [names]}

    blind: only what the blind tools try (rp._config.ls): no definitions tagged "example", and only the selected tags (rp.select_tags)"""
    return {k: sorted(n for names in _config.ls(k, path, blind=blind).items() if k != "header" or names[0] == "link" for n in names[1])
            for k in KINDS}

def load(ref: str, path=None, **params):
    """Load a link definition by "<kind>/<name>": `load("crc/crc32_mpeg2")`, `load("frame/hdlc")`, `load("header/ble_adv")`
    (config/<kind>/<name>.json; a link header is config/header/link/<name>.json), or below `path`. kind: crc | frame | header.
    A bare name works while one definition of the three kinds has it"""
    ref = str(ref).strip("/")
    kind, _, rest = ref.partition("/")
    kinds = [kind] if kind in KINDS and rest else list(KINDS)
    rest = rest if kind in KINDS and rest else ref
    hits = [(k, *_config.locate(k, rest, "link" if k == "header" else None, path)) for k in kinds]
    hits = [h for h in hits if h[1] is not None]
    if len(hits) > 1:
        raise ConfigError(f"{ref} is ambiguous: {[f'{k}/{rest}' for k, _, _ in hits]}")
    if not hits:
        raise FileNotFoundError(f"{ref}: no link definition (give <kind>/<name>, kind one of {KINDS}), have {ls(path)}")
    k, file, parts = hits[0]
    obj = _config.resolve(_config.read(file), params, f"{k}/{rest}")
    return _config.load_obj(k, parts[-1], obj)
