import numpy as np

from .modulation import Modulation

from .utils.constellation import Mapping, Points, Maps
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

    def __init__(self, mapping = None):
        super().__init__()
        self.constellation = Constellation(type(self).points, log=self.log)

        if mapping is not None:
            self.constellation.mapping =  Mapping(mapping)

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

    def set_mapping(self, mapping: Mapping):
        """Set modulation mapping"""
        self.constellation.mapping = mapping

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
