from ..constellation import ModConstellation
from ..utils import shapes

class QAM(ModConstellation):
    """Quadrature-amplitude modulation: `QAM(16)`. Square sizes (4, 16, 64, 256) are Gray; cross sizes (32, 128, 512)
    are as close to Gray as a cross allows"""
    def __init__(self, M=16, mapping=None):
        pts, lab = shapes.qam(M)
        points, maps = self._generated(pts, lab)
        super().__init__(mapping, points, maps)
        self.name = f"{M}QAM"
