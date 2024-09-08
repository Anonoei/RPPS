"""Modulation parent classes"""
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
    """Modulation Pipe"""
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
        """Set modulation mapping"""
        self.constellation.mapping = mapping

    def get_maps(self):
        """Get available maps"""
        return self.maps

    def init_meta(self, meta):
        """Initialize modulation meta"""
        if self.constellation.mapping is None:
            raise Exception("A mapping must be defined before modulating")
        from .meta import ModMeta
        meta.mod = ModMeta()
        meta.mod.fields["Name"] = type(self).name[:-3]
        meta.mod.fields["Type"] = type(self).name[-3:]
        meta.mod.fields["Map"] = self.constellation.mapping.str()

    def demodulate(self, syms: dobject.SymObject) -> dobject.ModData:
        """Convert IQ samples to bits"""
        ...

    def modulate(self, dobj: dobject.BitObject) -> dobject.SymData:
        """Convert bits to IQ samples"""
        ...

    def draw_refs(self, points: bool = True, ref: bool = True, ax=None):
        """Draw constellation points on viz"""
        ...

    @staticmethod
    def load(name, obj):
        """Load modulation from json"""
        ...

    def __rmatmul__(self, other):
        if isinstance(other, dobject.SymObject):
            return self.demodulate(other)
        elif isinstance(other, dobject.DataObject):
            return self.modulate(dobject.ensure_bit(other))
        raise TypeError(f"Cannot perform {type(self).__name__} on {type(other)}")


class PSK(Modulation):
    """Phase-shift keying parent"""

    def __str__(self):
        return f"{type(self).__name__}:{self.constellation.mapping.str()})"

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

    def demodulate(self, syms: dobject.SymData):
        data = self.constellation.demodulate(syms)
        return data

    def modulate(self, dobj: dobject.BitObject):
        syms = self.constellation.modulate(dobj)
        self.init_meta(dobj.meta)
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
    """Amplitude-shift keying parent"""


class FSK(Modulation):
    """Frequency-shift keying parent"""


class APSK(Modulation):
    """Amplitude-Phase-shift keying parent"""


class QAM(Modulation):
    """Quadrature-Amplitude modulation parent"""
