from ..constellation import ModConstellation
from ..utils import shapes

class ASK(ModConstellation):
    """Amplitude-shift keying: `ASK(2)` is on-off keying, `ASK(4)` has four amplitude levels (Gray)

    bipolar: levels are symmetric about zero (PAM, the default for M > 2) rather than 0..1
    """
    def __init__(self, M=2, mapping=None, bipolar=None):
        bipolar = M > 2 if bipolar is None else bipolar
        pts, lab = shapes.ask(M, bipolar)
        points, maps = self._generated(pts, lab)
        super().__init__(mapping, points, maps)
        self.name = f"{M}ASK" if M > 2 else "OOK"
