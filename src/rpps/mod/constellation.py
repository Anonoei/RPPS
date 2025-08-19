import numpy as np

from .modulation import Modulation

from .utils.constellation import Mapping, Points, Maps
from .utils.constellation import Constellation

from ..base import Mod, pproc, dtype

class ModConstellation(Modulation):
    """(De)mod using a constellation"""
    __slots__ = ("constellation", "mapping")
    name = "ModWithConstellation"
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
        sym = self.constellation.modulate(data.data)
        d = Mod(sym, pp=pproc.MOD, dt=dtype.SYMBOLS)
        return d
    def decode(self, data):
        codewords, distances = self.constellation.demodulate(data.data)
        d = Mod((codewords, distances), pp=pproc.MAP)
        return d

    def __radd__(self, other):
        data = super().__radd__(other)
        data.append((self.constellation.bits_per_symbol, 8))
        return data

    def __rsub__(self, other):
        data = super().__rsub__(other)
        data.append((8, self.constellation.bits_per_symbol))
        return data

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
