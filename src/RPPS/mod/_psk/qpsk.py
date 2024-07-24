from ..psk import PSK

from mod.constellation import Mapping, Maps, Points

class QPSK(PSK):
    points = Points(
        [
            .7 - .7j, -.7 - .7j,
            .7 + .7j, -.7 + .7j
        ]
    )
    maps = Maps(
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
