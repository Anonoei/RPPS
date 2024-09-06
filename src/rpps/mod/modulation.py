import numpy as np

import numpy as np
import matplotlib.pyplot as plt

from pyboiler.logger import Logger, Level

from . import Meta
from . import base
from . import dobject

from .constellation import Mapping, Points, Maps
from .constellation import Constellation


class Modulation(base.rpps.Pipe):
    __slots__ = ("log", "constellation", "mapping")
    name = "Modulation"
    points = Points([])
    maps = Maps([])

    def __init__(self, mapping = None):  # type: ignore
        self.log = Logger().Child("Modulation").Child(type(self).name)

        self.constellation = Constellation(type(self).points, log=self.log)

        if mapping is not None:
            self.constellation.mapping =  Mapping(mapping)

    def set_mapping(self, mapping: Mapping):
        self.constellation.mapping = mapping

    def get_maps(self):
        return self.maps

    def init_meta(self, meta):
        if self.constellation.mapping is None:
            raise Exception("A mapping must be defined before modulating")
        from .meta import ModMeta
        meta.mod = ModMeta()
        meta.mod.fields["Name"] = type(self).name[:-3]
        meta.mod.fields["Type"] = type(self).name[-3:]
        meta.mod.fields["Map"] = self.constellation.mapping.str()

    def demodulate(self, syms: dobject.SymData, meta=Meta()) -> dobject.ModData:
        ...

    def modulate(self, dobj: dobject.DataObject, meta=Meta()) -> dobject.SymData:
        ...

    def draw_refs(self, points: bool = True, ref: bool = True, ax=None):
        ...

    def __matmul__(self, other):
        print(f"Running matmul")
        if isinstance(other, dobject.SymData):
            return self.demodulate(other, other.meta)
        elif issubclass(type(other), dobject.DataObject):
            return self.modulate(other, other.meta)
        else:
            raise TypeError(f"Cannot perform {type(self).__name__} on {type(other)}")

    def __rmatmul__(self, other):
        print(f"Running rmatmul")
        if isinstance(other, dobject.SymData):
            return self.demodulate(other, other.meta)
        elif issubclass(type(other), dobject.DataObject):
            return self.modulate(other, other.meta)
        else:
            raise TypeError(f"Cannot perform {type(self).__name__} on {type(other)}")


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

    def demodulate(self, syms: dobject.SymData, meta=Meta()):
        data = self.constellation.demodulate(syms, meta)
        return data

    def modulate(self, dobj: dobject.DataObject, meta=Meta()):
        syms = self.constellation.modulate(dobj, meta)
        self.init_meta(meta)
        return syms

    @staticmethod
    def load(name, obj):
        def load_complex(comp):
            c = []
            for num in comp:
                c.append(num["real"] + num["imag"] * 1j)
            return c
        pnts = load_complex(obj["Points"])

        maps = [Mapping(m["map"], m["comment"]) for m in obj["Maps"]]

        impl = type(
            name,
            (PSK,),
            dict(name=name, points=Points(pnts), maps=Maps(maps))
        )()
        return impl


class ASK(Modulation):
    pass


class FSK(Modulation):
    pass


class APSK(Modulation):
    pass


class QAM(Modulation):
    pass
