import numpy as np

from ..constellation import ModConstellation, Mapping, Points, Maps

class PSK(ModConstellation):
    """Phase-shift keying parent"""
    @staticmethod
    def load(name, obj):
        # def load_complex(comp):
        #     c = []
        #     for num in comp:
        #         c.append(num["real"] + num["imag"] * 1j)
        #     return c
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
