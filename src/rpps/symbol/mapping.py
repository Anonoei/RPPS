"""Bit maps of a constellation: which value (bits, first bit most significant) each point carries, and the maps a receiver that
does not know the carrier phase or the IQ sense has to try (every rotation the constellation allows, with and without spectral inversion)"""
import re

import numpy as np

class Mapping:
    """Constellation map"""
    __slots__ = ("arr", "_comment", "_inv", "rotation", "turn", "aka", "tags")

    def __init__(self, map=None, comment="", turn=0.0, tags=()):
        self._comment = comment
        self.rotation = 0.0 # Radians the channel turns the constellation by for this map to be the right one
        self.turn = float(turn) # Radians the points sit turned by while this map is used (a standard's labelling of a turned set)
        self.aka = "" # A standard's name for this map, where one is given that is the same labels ("DVB-S2")
        self.tags = list(tags) # The standards that label the points by this map (a definition map's "tags")
        self.arr = np.array([], dtype=np.int64) if map is None else np.array(map)
        self._inv = False

    def copy(self):
        m = Mapping(self.arr.copy(), self._comment, self.turn, self.tags)
        m.rotation, m._inv, m.aka = self.rotation, self._inv, self.aka
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
        out = f"{self._comment} {'Inverted' if self._inv else 'Normal'}"
        return f"{out} ({self.aka})" if self.aka else out


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
    these is the map that decodes. Named R<degrees> <comment> Normal|Inverted. A later map with a name of its own (not
    R<degrees>: a standard's labelling, "DVB-S2") that is one of these gives it that name too ("R000 Grey Inverted
    (DVB-S2)"), so either picks it; one that is none of them (a map that turns the points) comes last, as given.
    Returns a new list of new maps
    """
    maps = [m.copy() for m in maps] # The caller's maps are not touched
    base = maps[0]
    if not re.match(r"^R\d+", base._comment):
        base._comment = f"R000 {base._comment}".strip() # The map as given is the unrotated one
    rest = re.sub(r"^R\d+(\.\d+)?\s*", "", base._comment).strip()
    named = {} # labels -> a map with a name of its own, the points unturned
    rest_named = [] # Those that turn the points: never one of the turned maps
    out, seen = [], {}
    for i, m in enumerate(maps):
        key = tuple(np.asarray(m.arr).tolist())
        if i and not re.match(r"^R\d+", m._comment):
            if m.turn:
                rest_named.append(m)
            elif key in seen:
                seen[key].aka, seen[key].tags = m._comment, [*seen[key].tags, *m.tags]
            else:
                named[key] = m
            continue
        out.append(m)
        seen[key] = m
    for th, flip, q in symmetries(points):
        arr = np.empty_like(np.asarray(base.arr))
        arr[q] = np.asarray(base.arr) # The point p moved to q, carrying its label
        key = tuple(arr.tolist())
        if key in seen:
            seen[key].rotation, seen[key].inverted = th, flip # A map that was already defined: record what it is
            continue
        deg = np.degrees(th)
        tag = f"R{deg:03.0f}" if abs(deg - round(deg)) < 1e-6 else f"R{deg:05.1f}"
        m = Mapping(arr, f"{tag} {rest}".strip(), base.turn) # (a turned map's turns use its points turned too)
        m.inverted = flip
        m.rotation = th
        if key in named:
            n = named.pop(key)
            m.aka, m.tags = n._comment, list(n.tags)
        seen[key] = m
        out.append(m)
    return out + list(named.values()) + rest_named

def hard_bits(y, pts, labels):
    """Hard bits of symbols y: the labels (values, first bit most significant) of the nearest points, as uint8 bits"""
    idx = np.argmin(np.abs(np.asarray(y)[:, None] - np.asarray(pts)[None, :]), axis=1)
    bps = int(np.log2(len(pts)))
    return ((np.asarray(labels, dtype=np.int64)[idx][:, None] >> np.arange(bps-1, -1, -1)) & 1).astype(np.uint8).reshape(-1)
