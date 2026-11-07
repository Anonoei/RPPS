import numpy as np

from .modulation import Modulation

from .utils.constellation import Mapping, Points, Maps, expand_maps
from .utils.constellation import Constellation

from ..base import Mod, pproc, dtype
from ..base.rpps import align

class ModConstellation(Modulation):
    """(De)mod using a constellation"""
    __slots__ = ("constellation", "mapping")
    name = "ModWithConstellation"
    maxlog = False # Max-log LLRs (nearest point) instead of the exact sum. The noise level is always estimated from the symbols
    points = Points([])
    maps = Maps([])

    def __init__(self, mapping = None, points=None, maps=None):
        super().__init__()
        if points is not None:
            self.points = points if isinstance(points, Points) else Points(points)
        if maps is not None:
            self.maps = maps if isinstance(maps, Maps) else Maps(list(maps))
        if len(self.maps) and not getattr(self.maps, "expanded", False):
            self.maps = Maps(expand_maps(self.points.arr, list(self.maps)), expanded=True) # Rotations and spectral inversions too
        self.constellation = Constellation(self.points, log=self.log)
        if mapping is None and len(self.maps):
            mapping = self.maps[0] # The first map unless told otherwise

        if mapping is not None:
            self.constellation.mapping = self._pick_map(mapping)

    @classmethod
    def _generated(cls, points, labels, comment="Gray"):
        """Points and one label list, as the pieces __init__ takes"""
        return Points(points), Maps([Mapping(np.asarray(labels), comment)])

    def __str__(self):
        return f"{type(self).__name__}:{self.constellation.mapping.str()}"

    def encode(self, data):
        ctx = dict(data.ctx)
        bits = align(ctx, (self, True), data.bits(), self.constellation.bits_per_symbol, pad=True)
        sym = self.constellation.modulate(bits) if len(bits) else np.zeros(0, dtype=dtype.SYMBOLS.dtype)
        return data.derive(sym, pproc.MOD, dtype.SYMBOLS, Mod, ctx)

    def decode(self, data):
        syms = data.data
        llr = self.constellation.demodulate(syms, maxlog=self.maxlog) if len(syms) else np.zeros(0, dtype=dtype.LLR.dtype)
        return data.derive(llr, pproc.MAP, dtype.LLR, Mod)

    def set_mapping(self, mapping):
        """Choose the bit mapping: an index into `get_maps()`, words of its name ("R090", "R090 inverted"), a `Mapping`, or a list of labels.
        Modulations start on their first map, so this is only needed for another one"""
        self.constellation.mapping = self._pick_map(mapping)
        return self

    def _pick_map(self, mapping):
        if isinstance(mapping, Mapping):
            return mapping
        if isinstance(mapping, (int, np.integer)):
            return self.maps[int(mapping)]
        if isinstance(mapping, str):
            words = mapping.lower().split() # Every word has to be in the name: "R090 Inverted"
            hits = [m for m in self.maps if all(w in m.comment.lower() for w in words)]
            if len(hits) > 1 and not {"normal", "inverted"} & set(words):
                hits = [m for m in hits if not m.inverted] # Unless it says inverted, the normal sense is meant
            if len(hits) != 1:
                raise ValueError(f"{len(hits)} maps match {mapping!r}, have {[m.comment for m in self.maps]}")
            return hits[0]
        return Mapping(mapping)

    def get_maps(self):
        """Get available maps"""
        return self.maps

    def draw_refs(self, points: bool = True, ref: bool = True, ax=None):
        import matplotlib.pyplot as plt
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot()
        if points:
            x = self.points.real()
            y = self.points.imag()
            ax.scatter(x=x, y=y, s=200, c="r")
            labels = self.constellation.mapping.arr
            # Add labels using annotate()
            for i, label in enumerate(labels):
                ax.annotate(
                    bin(label)[2:].zfill(self.constellation._bps),
                    (x[i], y[i]),
                    fontsize=20,
                )

        if ref:
            angle = np.linspace(0, 2 * np.pi, 150)
            radius = 1
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            ax.plot(x, y, "g")

    @staticmethod
    def load(name: str, obj: dict):
        raise NotImplementedError()
