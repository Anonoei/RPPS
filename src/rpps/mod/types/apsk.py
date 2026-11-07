from ..constellation import ModConstellation
from ..utils import shapes

class APSK(ModConstellation):
    """Amplitude-phase-shift keying on rings: `APSK(16)` is 4+12, `APSK(32)` is 4+12+16. Or give the rings: `APSK(rings=(8, 8))`

    ratios: radius of the outer rings over the innermost
    """
    RINGS = {16: (4, 12), 32: (4, 12, 16), 8: (1, 7), 64: (4, 12, 20, 28)}
    def __init__(self, M=16, mapping=None, rings=None, ratios=None):
        rings = self.RINGS[M] if rings is None else tuple(rings)
        pts, lab = shapes.apsk(rings, ratios)
        points, maps = self._generated(pts, lab, "ring + Gray")
        super().__init__(mapping, points, maps)
        self.name = f"{sum(rings)}APSK"
