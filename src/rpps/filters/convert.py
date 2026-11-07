"""Bring filter coefficients from elsewhere into the config format

A definition in config/ never refers to another file; this is how coefficients that live in a file (or in another tool's output)
become one. Read a file, get the definition as a dict, or have it written into a config directory:

    d = rp.filters.import_fir("taps.txt")                                  # {"base": "fir", "type": "taps", "taps": [...]}
    rp.filters.import_fir("lp.f32", dtype="f32", name="my_lp", dest="mine")  # writes mine/filters/fir/taps/my_lp.json
    rp.filters.import_iir("sections.npy", form="sos", name="my_iir")
    rp.add_config_path("mine"); rp.filters.load("fir", "taps/my_lp")

Formats: text/CSV, NumPy .npy/.npz, headerless raw binary (GNU Radio file sinks, `tofile`), numbers in a C/Python/Verilog array
literal, Xilinx .coe. Arrays already in memory (scipy, firdes...) go through `fir_def`, `sos_def`, `ba_def` and `zpk_def`.
"""
from pathlib import Path
import re

import numpy as np

from .. import _config
from . import iir as _iir

MAX_BYTES = 64 * 2**20 # Coefficient files are small: anything bigger is the wrong file

DTYPES = {"f32": "<f4", "f64": "<f8", "cf32": "<c8", "cf64": "<c16", "i8": "i1", "u8": "u1", "i16": "<i2", "i32": "<i4"}
EXTENSIONS = {".npy": "npy", ".npz": "npz", ".coe": "coe",
              ".txt": "text", ".csv": "text", ".tsv": "text", ".dat": "text", ".taps": "text",
              ".h": "source", ".c": "source", ".cc": "source", ".cpp": "source", ".hpp": "source", ".py": "source",
              ".v": "source", ".vh": "source", ".sv": "source", ".vhd": "source",
              ".bin": "raw", ".raw": "raw", ".f32": "raw", ".f64": "raw", ".cf32": "raw", ".cf64": "raw", ".fc32": "raw"}
IMPLIED_DTYPE = {".f32": "f32", ".f64": "f64", ".cf32": "cf32", ".fc32": "cf32", ".cf64": "cf64"}

# ---- Reading: a file becomes a table, a list of rows (1-D arrays)

def _numbers(tokens, where):
    out = []
    for t in tokens:
        try:
            out.append(complex(re.sub(r"(?<=\d)[iI]$", "j", t)))
        except ValueError:
            raise ValueError(f"{where}: {t!r} is not a number") from None
    arr = np.array(out, dtype=complex)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{where}: inf or nan in the coefficients")
    return arr.real if not np.any(arr.imag) else arr

def _read_text(text, name):
    rows = []
    for i, line in enumerate(text.splitlines(), 1):
        line = re.split(r"#|%|//", line)[0]
        tokens = [t for t in re.split(r"[\s,;]+", line.strip()) if t]
        if tokens:
            rows.append(_numbers(tokens, f"{name}:{i}"))
    return rows

def _block(text):
    """The first brace or bracket group after the first "=" (or the whole text), as nested lists of number text"""
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)
    text = "\n".join(re.split(r"//|#", ln)[0] for ln in text.splitlines())
    start = text.find("=") + 1
    m = re.search(r"[{\[]", text[start:])
    if not m:
        raise ValueError("no array literal ({...} or [...]) found")
    i = start + m.start()
    depth, rows, cur = 0, [], ""
    for ch in text[i:]:
        if ch in "{[":
            depth += 1
            cur = ""
        elif ch in "}]":
            if cur.strip():
                rows.append(cur)
            cur = ""
            depth -= 1
            if depth == 0:
                break
        else:
            cur += ch
    if depth != 0:
        raise ValueError("unbalanced brackets in the array literal")
    return rows

def _read_source(text, name):
    rows = []
    for k, row in enumerate(_block(text), 1):
        tokens = [re.sub(r"(?<=[\d.])[fFlLuU]+$", "", t) for t in re.split(r"[\s,;]+", row.strip()) if t]
        rows.append(_numbers(tokens, f"{name}: array row {k}"))
    return rows

def _read_coe(text, name):
    lines = [ln for ln in text.splitlines() if not ln.strip().startswith(";")]
    fields = {}
    for stmt in "\n".join(lines).split(";"):
        if "=" in stmt:
            k, _, v = stmt.partition("=")
            fields[k.strip().lower()] = v.strip()
    radix = fields.get("radix", fields.get("memory_initialization_radix", "10"))
    if radix != "10":
        raise ValueError(f"{name}: radix {radix} .coe files are not supported (only decimal)")
    for key in ("coefdata", "coefficient_vector", "memory_initialization_vector"):
        if key in fields:
            return [_numbers([t for t in re.split(r"[\s,]+", fields[key]) if t], f"{name}: {key}")]
    raise ValueError(f"{name}: no coefdata / coefficient_vector in the .coe file")

def read_table(path, fmt=None, dtype=None, key=None):
    """The numbers of a file as a list of 1-D arrays (one per row of text, one for a vector).

    fmt: text, npy, npz, raw, source, coe (default from the extension). dtype: for raw files, f32 f64 cf32 cf64 i8 u8 i16 i32 or a
    numpy dtype string. key: the array to take from an .npz"""
    path = Path(path)
    if path.stat().st_size > MAX_BYTES:
        raise ValueError(f"{path.name} is {path.stat().st_size} bytes: that is not a coefficient file")
    ext = path.suffix.lower()
    fmt = fmt or EXTENSIONS.get(ext)
    if fmt is None:
        raise ValueError(f"{path.name}: unknown file type, say fmt= one of text, npy, npz, raw, source, coe")
    if fmt == "text":
        return _read_text(path.read_text(), path.name)
    if fmt == "source":
        return _read_source(path.read_text(), path.name)
    if fmt == "coe":
        return _read_coe(path.read_text(), path.name)
    if fmt == "raw":
        dtype = dtype or IMPLIED_DTYPE.get(ext)
        if dtype is None:
            raise ValueError(f"{path.name}: a raw file has no header, say dtype= (f32, f64, cf32, cf64, i16...)")
        dt = np.dtype(DTYPES.get(dtype, dtype))
        if path.stat().st_size % dt.itemsize:
            raise ValueError(f"{path.name}: {path.stat().st_size} bytes is not a whole number of {dtype} values")
        return [np.fromfile(path, dtype=dt)]
    if fmt in ("npy", "npz"):
        loaded = np.load(path, allow_pickle=False)
        if fmt == "npz":
            names = list(loaded.files)
            if key is None and len(names) != 1:
                raise ValueError(f"{path.name} holds {names}: say key=")
            if key is not None and key not in names:
                raise ValueError(f"{path.name} has no array {key!r}, only {names}")
            loaded = loaded[key or names[0]]
        a = np.asarray(loaded)
        return [a] if a.ndim <= 1 else [row for row in a]
    raise ValueError(f"unknown format {fmt!r}")

def read_array(path, fmt=None, dtype=None, key=None):
    """The numbers of a file as one array: 1-D for a vector or a single row, 2-D for rows of one length"""
    rows = read_table(path, fmt, dtype, key)
    if len(rows) == 1:
        return np.asarray(rows[0])
    if len({len(r) for r in rows}) != 1:
        raise ValueError(f"{Path(path).name}: rows of different lengths {[len(r) for r in rows]}")
    return np.vstack(rows)

# ---- Definitions from arrays

def _nums(a):
    """A real array as a (nested) list of JSON numbers"""
    a = np.asarray(a)
    shape = a.shape
    flat = a.ravel()
    vals = [float(str(np.float32(v))) for v in flat] if a.dtype == np.float32 else [float(v) for v in flat]
    return np.asarray(vals).reshape(shape).tolist()

def _real(x):
    """A float array, keeping the precision of float32 data"""
    a = np.atleast_1d(np.asarray(x))
    return a if a.dtype.kind == "f" else a.astype(float)

def _fields(kind, fields):
    return {"comment": fields.pop("comment", None) or f"Imported {kind}", **fields}

def fir_def(taps, **fields):
    """Definition of an FIR with these taps (real, or complex: the imaginary parts go in `taps_imag`). fields: comment, gain, fs..."""
    t = np.asarray(taps)
    if t.ndim != 1 or t.size == 0:
        raise ValueError(f"taps must be a non-empty 1-D list, got shape {t.shape}")
    if not np.all(np.isfinite(t)):
        raise ValueError("taps contain inf or nan")
    obj = {"base": "fir", "type": "taps", **_fields(f"{t.size}-tap FIR", dict(fields))}
    if t.dtype.kind == "c" and np.any(t.imag):
        obj["taps"], obj["taps_imag"] = _nums(t.real), _nums(t.imag)
    else:
        obj["taps"] = _nums(t.real if t.dtype.kind == "c" else t)
    return obj

def _check_stable(sos):
    _iir.IIR(sos) # Raises on an unstable pole

def sos_def(sos, **fields):
    """Definition of an IIR from second-order sections, rows [b0 b1 b2 a0 a1 a2] (scipy's `sos`)"""
    s = _iir.as_sos(sos)
    _check_stable(s)
    return {"base": "iir", "type": "sos", **_fields(f"{len(s)}-section IIR", dict(fields)), "sos": _nums(_real(sos).reshape(-1, 6))}

CONVENTIONS = {"scipy": False, "matlab": False, "gnuradio-new": False, "gnuradio": True, "gnuradio-old": True}

def ba_def(b, a, convention="scipy", **fields):
    """Definition of an IIR from a transfer function. convention (the sign of the feedback taps `a`):
        scipy, matlab, gnuradio-new   y[n] = sum b[k] x[n-k] - sum a[k] y[n-k] (k >= 1), a[0] = 1
        gnuradio (gnuradio-old)       y[n] = sum b[k] x[n-k] + sum a[k] y[n-k] (k >= 1); a[0] is ignored (iir_filter_ffd oldstyle=True)
    The definition holds the scipy form."""
    if convention not in CONVENTIONS:
        raise ValueError(f"convention must be one of {sorted(CONVENTIONS)}, not {convention!r}")
    b, a = _real(b), _real(a)
    if CONVENTIONS[convention]:
        a = np.concatenate(([1.0], -a[1:]))
    _check_stable(_iir.tf2sos(b, a))
    return {"base": "iir", "type": "ba", **_fields(f"order {max(len(a), len(b)) - 1} IIR", dict(fields)), "b": _nums(b), "a": _nums(a)}

def zpk_def(z, p, k, **fields):
    """Definition of an IIR from zeros, poles and gain (z-plane; complex values are stored as [re, im])"""
    z, p = np.atleast_1d(np.asarray(z, dtype=complex)), np.atleast_1d(np.asarray(p, dtype=complex))
    _check_stable(_iir.zpk2sos(z, p, k))
    pair = lambda v: [[float(x.real), float(x.imag)] if x.imag else float(x.real) for x in v]
    return {"base": "iir", "type": "zpk", **_fields(f"{len(p)}-pole IIR", dict(fields)), "z": pair(z), "p": pair(p), "k": float(k)}

# ---- From files

def import_fir(path, fmt=None, dtype=None, key=None, scale=1.0, name=None, dest=None, overwrite=False, **fields):
    """The FIR in a file as a config definition (a dict). A text file may wrap the taps over lines. scale multiplies the taps (integer .coe data: 1/32768 for 16 bit).

    name: also write it as config/filters/fir/taps/<name>.json under dest (default: the first added config path, else the shipped
    config; an existing file is kept unless overwrite) and return the file. fields: comment, gain..."""
    rows = read_table(path, fmt, dtype, key)
    if (fmt or EXTENSIONS.get(Path(path).suffix.lower())) == "text": # Taps wrapped over several lines, or one per line
        rows = [np.concatenate(rows)] if rows else rows
    if len(rows) != 1:
        raise ValueError(f"{Path(path).name} has {len(rows)} rows: taps are one list (an sos or ba table is for import_iir)")
    obj = fir_def(np.asarray(rows[0]) * scale, **{"comment": f"FIR from {Path(path).name}", **fields})
    return obj if name is None else _config.save("filters", "fir", f"taps/{name}", obj, dest, overwrite)

def import_iir(path, form="sos", fmt=None, dtype=None, key=None, convention="scipy", name=None, dest=None, overwrite=False, **fields):
    """The IIR in a file as a config definition. form says what the numbers are:
        sos   rows of 6 (b0 b1 b2 a0 a1 a2), one per section
        ba    two rows or arrays, b then a (see `ba_def` for convention)
        zpk   three rows, zeros, poles, then the gain; from an .npz, the arrays z, p, k
    A .npz may hold the arrays by name (sos | b, a | z, p, k). name/dest/overwrite as for `import_fir` (written under filters/iir/<form>/)."""
    fields = {"comment": f"IIR from {Path(path).name}", **fields}
    fmt = fmt or EXTENSIONS.get(Path(path).suffix.lower())
    if fmt == "npz":
        arrays = dict(np.load(path, allow_pickle=False))
        names = {"sos": ("sos",), "ba": ("b", "a"), "zpk": ("z", "p", "k")}.get(form)
        if names is None or not all(n in arrays for n in names):
            raise ValueError(f"{Path(path).name} holds {sorted(arrays)}: form={form!r} needs {names}")
        rows = [np.atleast_1d(arrays[n]) for n in names] if form != "sos" else [r for r in np.atleast_2d(arrays["sos"])]
    else:
        rows = read_table(path, fmt, dtype, key)
    if form == "sos":
        obj = sos_def(np.vstack(rows) if len({len(r) for r in rows}) == 1 else rows, **fields)
    elif form == "ba":
        if len(rows) != 2:
            raise ValueError(f"form='ba' needs two rows (b then a), {Path(path).name} has {len(rows)}")
        obj = ba_def(rows[0], rows[1], convention, **fields)
    elif form == "zpk":
        if len(rows) != 3 or len(rows[2]) != 1:
            raise ValueError(f"form='zpk' needs three rows (zeros, poles, then one gain), {Path(path).name} has {[len(r) for r in rows]}")
        obj = zpk_def(rows[0], rows[1], rows[2][0].real, **fields)
    else:
        raise ValueError(f"form must be sos, ba or zpk, not {form!r}")
    return obj if name is None else _config.save("filters", "iir", f"{form}/{name}", obj, dest, overwrite)
