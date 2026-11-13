"""Bit maps of a constellation: which value (bits, first bit most significant) each point carries, and the maps a receiver that
does not know the carrier phase or the IQ sense has to try (every rotation the constellation allows, with and without spectral inversion)"""
import re

import numpy as np

class Mapping:
    """Constellation map"""
    __slots__ = ("arr", "_comment", "_inv", "rotation")

    def __init__(self, map=None, comment=""):
        self._comment = comment
        self.rotation = 0.0 # Radians the channel turns the constellation by for this map to be the right one
        self.arr = np.array([], dtype=np.int64) if map is None else np.array(map)
        self._inv = False

    def copy(self):
        m = Mapping(self.arr.copy(), self._comment)
        m.rotation, m._inv = self.rotation, self._inv
        return m

    def str(self):
        """Return map values as str"""
        return "-".join(self.arr.astype(str))

    def __len__(self):
        return len(self.arr)

    def __str__(self):
        return str(self.arr)

    def __repr__(self):
        return f"({str(self)}, {self.comment})"

    def __getitem__(self, item):
        return self.arr[item]

    def __setitem__(self, item, val):
        self.arr[item] = val

    @property
    def inverted(self):
        return self._inv

    @inverted.setter
    def inverted(self, val: bool):
        self._inv = val

    @property
    def comment(self):
        if self._inv:
            return f"{self._comment} Inverted"
        return f"{self._comment} Normal"


def _perm(points, fn):
    """Where each point lands under fn (a transform of the plane), or None if some point doesn't land on a point"""
    pts = np.asarray(points, dtype=np.complex128)
    new = fn(pts)
    scale = max(np.max(np.abs(pts)), 1e-12)
    d = np.abs(new[:, None] - pts[None, :])
    q = np.argmin(d, axis=1)
    if np.max(d[np.arange(len(pts)), q]) > 1e-6*scale or len(set(q.tolist())) != len(pts):
        return None
    return q

def symmetries(points):
    """Every (rotation [rad], conjugated) that maps the constellation onto itself, as (angle, flip, permutation)
    A flip (spectral inversion) is applied first, then the rotation."""
    pts = np.asarray(points, dtype=np.complex128)
    ref = next((p for p in pts if abs(p) > 1e-9), None)
    out = []
    for flip in (False, True):
        base = np.conj if flip else (lambda z: z)
        angles = {0.0}
        if ref is not None:
            for p in pts:
                if abs(abs(p) - abs(ref)) < 1e-6*abs(ref):
                    angles.add(float(np.angle(p/ref) % (2*np.pi)) if not flip else float(np.angle(p/np.conj(ref)) % (2*np.pi)))
        for th in sorted(angles):
            q = _perm(pts, lambda z, th=th, base=base: base(z)*np.exp(1j*th))
            if q is not None:
                out.append((th, flip, q))
    return out

def expand_maps(points, maps):
    """The given maps, then the first one turned by every rotation the constellation allows and spectrally inverted
    (conjugated) with each. The point is a receiver that doesn't know the carrier phase or the IQ sense: one of
    these is the map that decodes. Named R<degrees> <comment> Normal|Inverted. Returns a new list of new maps
    """
    maps = [m.copy() for m in maps] # The caller's maps are not touched
    base = maps[0]
    if not re.match(r"^R\d+", base._comment):
        base._comment = f"R000 {base._comment}".strip() # The map as given is the unrotated one
    rest = re.sub(r"^R\d+(\.\d+)?\s*", "", base._comment).strip()
    out, seen = [], {}
    for m in maps:
        out.append(m)
        seen[tuple(np.asarray(m.arr).tolist())] = m
    for th, flip, q in symmetries(points):
        arr = np.empty_like(np.asarray(base.arr))
        arr[q] = np.asarray(base.arr) # The point p moved to q, carrying its label
        key = tuple(arr.tolist())
        if key in seen:
            seen[key].rotation, seen[key].inverted = th, flip # A map that was already defined: record what it is
            continue
        deg = np.degrees(th)
        tag = f"R{deg:03.0f}" if abs(deg - round(deg)) < 1e-6 else f"R{deg:05.1f}"
        m = Mapping(arr, f"{tag} {rest}".strip())
        m.inverted = flip
        m.rotation = th
        seen[key] = m
        out.append(m)
    return out
