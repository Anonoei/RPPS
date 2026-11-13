"""LDPC parity-check matrices in MacKay's alist format (what GNU Radio's gr-fec and most LDPC tools read and write)

    rp.code.import_alist("code.alist")                          the definition, as a dict (`rp.code.init("mine", d)` builds it)
    rp.code.import_alist("code.alist", "mine", dest="my_config")   writes my_config/fec/block/ldpc/mine.json, `add_config_path("my_config")`
"""
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
    name: also write it as config/fec/block/ldpc/<name>.json, under `dest` (default: the first added config path, else the shipped config);
    an existing file is not replaced. Returns the file"""
    n, checks = read_alist(path)
    obj = {"base": "block", "type": "ldpc", "comment": f"LDPC code from {Path(path).name}", **fields, "n": n, "checks": checks}
    if name is None:
        return obj
    return _config.save("fec", "block", f"ldpc/{name}", obj, dest)
