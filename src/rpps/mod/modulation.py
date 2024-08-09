import numpy as np

import numpy as np
import matplotlib.pyplot as plt

from pyboiler.logger import Logger, Level

from . import Meta
from . import Stream

from .constellation import Mapping, Points
from .constellation import Constellation


class Modulation:
    __slots__ = ("log", "constellation", "mapping")
    name = "Modulation"
    points = Points([])

    def __init__(self, mapping = None):  # type: ignore
        self.log = Logger().Child("Modulation").Child(type(self).name)

        self.constellation = Constellation(type(self).points, log=self.log)

        if mapping is not None:
            self.constellation.mapping =  Mapping(mapping)

    def set_mapping(self, mapping: Mapping):
        self.constellation.mapping = mapping

    def init_meta(self, meta):
        if self.constellation.mapping is None:
            raise Exception("A mapping must be defined before modulating")
        from .meta import ModMeta
        meta.mod = ModMeta()
        meta.mod.fields["Name"] = type(self).name[:-3]
        meta.mod.fields["Type"] = type(self).name[-3:]
        meta.mod.fields["Map"] = self.constellation.mapping.str()

    def demodulate(self, symbols: np.ndarray, meta=Meta()):
        ...

    def modulate(self, data: Stream, meta=Meta()):
        ...

    def draw_refs(self, points: bool = True, ref: bool = True, ax=None):
        ...


class PSK(Modulation):

    def draw_refs(self, points: bool = True, ref: bool = True, ax=None):
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

    def demodulate(self, symbols: np.ndarray, meta=Meta()):
        data = self.constellation.demodulate(symbols, meta)

        return data, meta

    def modulate(self, data: Stream, meta=Meta()):
        symbols = self.constellation.modulate(data, meta)

        self.init_meta(meta)
        return symbols, meta


class ASK(Modulation):
    pass


class FSK(Modulation):
    pass


class APSK(Modulation):
    pass


class QAM(Modulation):
    pass
