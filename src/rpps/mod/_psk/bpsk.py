from ..psk import PSK

from ..constellation import Mapping, Maps, Points

class BPSK(PSK):
    name = "BPSK"
    points = Points(
        [
            1 + 0j, -1 + 0j,
        ]
    )
    maps = Maps(
        [
            Mapping([
                0, 1,
            ]),
            Mapping([
                1, 0
            ]),
        ]
    )
