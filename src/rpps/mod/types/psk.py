import numpy as np

from ..constellation import ModConstellation, Mapping, Points, Maps
from ..utils import shapes

class PSK(ModConstellation):
    """Phase-shift keying: `PSK(8)` makes M points on the unit circle with a Gray map. Definitions in config/mod/psk
    (`rp.mod.load("QPSK")`) give the DVB and other specific maps"""
    def __init__(self, M=None, mapping=None, offset=None):
        if M is None:
            super().__init__(mapping)
        else:
            pts, lab = shapes.psk(M, offset)
            points, maps = self._generated(pts, lab)
            super().__init__(mapping, points, maps)
            self.name = f"{M}PSK"

    @staticmethod
    def load(name, obj):
        def load_magpha(comp):
            c = []
            for num in comp:
                c.append(num["mag"] * np.exp(1j * np.deg2rad(num["pha"])))
            return c

        pnts = load_magpha(obj["Points"])
        maps = [Mapping(m["map"], m["comment"]) for m in obj["Maps"]]

        impl = type(name, (PSK,),
            dict(name=name, points=Points(pnts), maps=Maps(maps))
        )()
        return impl
