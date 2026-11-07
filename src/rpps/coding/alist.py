"""LDPC parity-check matrices in MacKay's alist format (what GNU Radio's gr-fec and most LDPC tools read and write)

    rp.coding.import_alist("code.alist")                          the definition, as a dict (`rp.coding.init("mine", d)` builds it)
    rp.coding.import_alist("code.alist", "mine", dest="my_config")   writes my_config/coding/block/ldpc/mine.json, `add_config_path("my_config")`
"""
import json
from pathlib import Path

from .. import _config

def read_alist(path):
    """(n, checks): code length and, for each parity check, the bit positions (0-based) it covers"""
    lines = [ln.split() for ln in Path(path).read_text().splitlines() if ln.strip()]
    n, m = int(lines[0][0]), int(lines[0][1])
    if len(lines) < 4 + n + m:
        raise ValueError(f"{path}: an alist for {n} bits and {m} checks has {4 + n + m} lines, found {len(lines)}")
    cols = [[int(x) - 1 for x in ln if int(x) > 0] for ln in lines[4:4 + n]] # Zeros pad the rows to the longest
    rows = [[int(x) - 1 for x in ln if int(x) > 0] for ln in lines[4 + n:4 + n + m]]
    from_cols = [[] for _ in range(m)]
    for j, c in enumerate(cols):
        for i in c:
            from_cols[i].append(j)
    if [sorted(r) for r in rows] != from_cols:
        raise ValueError(f"{path}: the row lines and the column lines describe different matrices")
    return n, rows

def import_alist(path, name=None, dest=None, **fields):
    """Definition of the LDPC code in an alist file. fields: more keys for it (`comment`, `k` + `systematic`, `iters`...).

    Without `k`, the message positions come from GF(2) elimination of H, which only suits small codes: give `k` and `systematic`
    ("head": message first, "tail", or a list of positions) for big ones.
    name: also write it as config/coding/block/ldpc/<name>.json, under `dest` (default: the highest config layer). Returns the file"""
    n, checks = read_alist(path)
    obj = {"base": "block", "type": "ldpc", "comment": f"LDPC code from {Path(path).name}", **fields, "n": n, "checks": checks}
    if name is None:
        return obj
    base = Path(dest) if dest else _config.find_config("coding").parent
    file = base / "coding" / "block" / "ldpc" / f"{name.lower()}.json"
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(json.dumps(obj, separators=(",", ":")))
    return file
