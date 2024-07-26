from ..psk import PSK

from ..constellation import Mapping, Maps, Points

class EPSK(PSK):
    name = "8PSK"
    points = Points(
        [
                      0 - 1j,
              .7 - .7j,   -.7 -.7j,
            1, + 0j          -1 + 0j,
              .7 + .7j,   -.7 + .7j,
                      0 + 1j
        ]
    )
    maps = Maps(
        [
            Mapping([
                     2,
                  3,    6,
                1,        7,
                  0,    5,
                     4
            ]),
        ]
    )
