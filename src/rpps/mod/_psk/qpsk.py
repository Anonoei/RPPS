from ..modulation import PSK

from ..constellation import Mapping, Maps, Points


QPSK_Maps = Maps(
        [
            Mapping([
                0, 1,
                2, 3,
            ]),
            Mapping([
                0, 2,
                1, 3,
            ]),
            Mapping([
                3, 1,
                2, 0,
            ]),
            Mapping([
                3, 2,
                1, 0,
            ]),
        ]
    )

class QPSK(PSK):
    name = "QPSK"
    points = Points(
        [
            .7 - .7j, -.7 - .7j,
            .7 + .7j, -.7 + .7j
        ]
    )
