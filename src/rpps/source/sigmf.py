"""SigMF recordings: the samples and what they were recorded with (sigmf.org, the core namespace)

    src = rp.source.SigMF("capture.sigmf-meta")          # or the .sigmf-data path: Fs, cf and the format from the metadata
    for out in src(65536) + rx: ...
    src.annotations                                        # [{"core:sample_start": ..., "core:label": ...}, ...]

    with rp.source.SigMFSink("out", Fs=2e6, cf=915e6, datatype="cf32_le") as sink:
        blocks >> sink                                     # out.sigmf-data, and out.sigmf-meta when it is closed

Datatypes: complex or real, float (f32, f64) or integer (i8, i16, i32, u8, u16, u32), little or big endian
("cf32_le", "ci16_be", "ri8"). Integers are scaled to [-1, 1) as File's are. Blocks start at the recording's
`core:sample_start` of its first capture; a capture's `core:frequency` sets cf (the first; later captures are listed in
`captures`). numpy and json only: no sigmf package needed. How the radio received (pysdrlib's `pysdrlib:rx`: the chain
and its LO, whether it inverts, the reference) is the source's `context` (see rp.source.Source: an inverting chain is
undone as the samples are read).
"""
import json
from pathlib import Path

import numpy as np

from .source import Source

def dtype_of(datatype):
    """numpy dtype of the real components, and whether samples are complex, from a SigMF core:datatype"""
    d = datatype.lower()
    cplx = d.startswith("c")
    if not (cplx or d.startswith("r")):
        raise ValueError(f"SigMF datatype {datatype!r}: starts with c (complex) or r (real)")
    body, _, endian = d[1:].partition("_")
    kind, bits = body[0], int(body[1:])
    if kind not in "fiu" or bits not in (8, 16, 32, 64) or (kind == "f" and bits not in (32, 64)):
        raise ValueError(f"SigMF datatype {datatype!r}: not a supported type")
    order = "<" if endian in ("", "le") else ">" if endian == "be" else None
    if order is None:
        raise ValueError(f"SigMF datatype {datatype!r}: endianness le or be")
    return np.dtype(f"{order}{kind}{bits//8}"), cplx

def _paths(path):
    p = Path(path)
    stem = p.with_suffix("") if p.suffix in (".sigmf-meta", ".sigmf-data", ".sigmf") else p
    return Path(str(stem) + ".sigmf-meta"), Path(str(stem) + ".sigmf-data")

def _context(glob):
    """The recording's pysdrlib RxContext (`pysdrlib:rx`), None when it has none (or pysdrlib can't read it)"""
    obj = glob.get("pysdrlib:rx")
    if obj is None:
        return None
    try:
        from pysdrlib.base.stream import RxContext
    except ImportError: # pysdrlib before 0.5.1, or none: the samples are read as they are
        return None
    return RxContext.from_json(obj)

class SigMF(Source):
    """A SigMF recording (the -meta or -data path, or their stem): Fs, cf and the datatype from its metadata

    The data file stays memory mapped until `close()` (or the end of a `with` block): close it before the file is deleted
    or overwritten, which Windows refuses while it is mapped"""
    def __init__(self, path, sps=0.0, scale=True, correct=True):
        self.meta_path, self.data_path = _paths(path)
        self.meta = json.loads(self.meta_path.read_text())
        g = self.meta.get("global", {})
        self.captures = self.meta.get("captures", [])
        self.annotations = self.meta.get("annotations", [])
        first = self.captures[0] if self.captures else {}
        super().__init__(float(g.get("core:sample_rate", 1.0)), float(first.get("core:frequency", 0.0)), sps,
                         context=_context(g), correct=correct)
        self.datatype = g["core:datatype"]
        self.dtype, self.complex = dtype_of(self.datatype)
        self.scale = (1.0/float(2**(8*self.dtype.itemsize - 1))) if (scale and self.dtype.kind in "iu") else 1.0
        self.start = int(first.get("core:sample_start", 0))
        self._map = None
        self.open()

    @property
    def data(self):
        """The recording's raw values, memory mapped (mapped again if the source was closed)"""
        self.open()
        return self._map

    def open(self):
        if self._map is None:
            self._map = np.memmap(self.data_path, dtype=self.dtype, mode="r")

    def close(self):
        """Unmap the data file. Windows can't delete or replace a file while it is mapped; reading again maps it again"""
        self._map = None # The blocks are copies, so this was the only reference

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *a):
        self.close()

    def __str__(self):
        return f"SigMF({self.data_path.name}, {self.datatype}, {self.Fs/1e6:g} MS/s, cf {self.cf/1e6:g} MHz)"

    def __len__(self):
        return len(self.data)//(2 if self.complex else 1) - self.start

    def _read(self, a, b):
        """Samples a .. b (from the first capture's start) as floats"""
        w = 2 if self.complex else 1
        raw = np.asarray(self.data[(self.start + a)*w:(self.start + b)*w], dtype=np.float64)
        if self.dtype.kind == "u":
            raw = raw - float(2**(8*self.dtype.itemsize - 1))
        raw = raw*self.scale
        if self.complex:
            return (raw[0::2] + 1j*raw[1::2]).astype(np.complex64 if self.dtype.itemsize < 8 else np.complex128)
        return raw.astype(np.float32 if self.dtype.itemsize < 8 else np.float64)

    def peek(self, count, start=0):
        return self._sense(self._read(int(start), min(int(start) + int(count), len(self))))

    def _samples(self, count):
        count, n = int(count), len(self)
        for i in range(0, n, count):
            yield self._block(self._read(i, min(i + count, n)), flush=i + count >= n)

class SigMFSink:
    """Blocks written as a SigMF recording: `path`.sigmf-data as they come, `path`.sigmf-meta when closed

    datatype: the SigMF core:datatype written ("cf32_le" default; integer types are scaled from [-1, 1) and clipped).
    Fs, cf: the recording's; else the first block's. annotations: added to the metadata. context: how the samples were
    received (a pysdrlib RxContext, kept as `pysdrlib:rx`); else the first block's stream's (`stream.rx`), marked undone
    when its source conjugated them"""
    def __init__(self, path, Fs=None, cf=None, datatype="cf32_le", description=None, annotations=(), context=None):
        self.meta_path, self.data_path = _paths(path)
        self.datatype, self.Fs, self.cf, self.description = datatype, Fs, cf, description
        self.context = context
        self.dtype, self.complex = dtype_of(datatype)
        self.annotations = list(annotations)
        self.count = 0
        self._f = open(self.data_path, "wb")

    def write(self, data):
        from ..base.data import _Data
        if isinstance(data, _Data):
            if self.Fs is None:
                self.Fs = data.Fs
            if self.cf is None:
                self.cf = data.cf
            if self.context is None and data.meta.stream.rx is not None: # How the source received: kept, and whether
                st = data.meta.stream                                     # its samples were put right (`undone`)
                self.context = st.rx
                if st.rx_conjugated and not getattr(st.rx, "undone", False):
                    import dataclasses
                    self.context = dataclasses.replace(st.rx, undone=True)
        x = np.asarray(data.data if isinstance(data, _Data) else data)
        parts = np.stack((x.real, x.imag), axis=-1).reshape(-1) if self.complex else np.real(x).reshape(-1)
        if self.dtype.kind in "iu":
            full = float(2**(8*self.dtype.itemsize - 1))
            parts = np.clip(np.round(parts*full + (full if self.dtype.kind == "u" else 0.0)), np.iinfo(self.dtype).min, np.iinfo(self.dtype).max)
        self._f.write(parts.astype(self.dtype).tobytes())
        self.count += len(x)
        return self

    def close(self):
        if self._f is None:
            return
        self._f.close()
        self._f = None
        g = {"core:datatype": self.datatype, "core:sample_rate": float(self.Fs or 1.0), "core:version": "1.0.0",
             "core:recorder": "rpps"}
        if self.description:
            g["core:description"] = self.description
        if self.context is not None:
            g["core:extensions"] = [{"name": "pysdrlib", "version": "1.0.0", "optional": True}]
            g["pysdrlib:rx"] = self.context.json()
        meta = {"global": g, "captures": [{"core:sample_start": 0, "core:frequency": float(self.cf or 0.0)}],
                "annotations": self.annotations}
        self.meta_path.write_text(json.dumps(meta, indent=2) + "\n")

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()
