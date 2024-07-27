from ..modulation import PSK

from ..constellation import Mapping, Maps, Points

BPSK_Maps = Maps(
        [
            Mapping([
                0, 1,
            ]),
            Mapping([
                1, 0
            ]),
        ]
    )

class BPSK(PSK):
    name = "BPSK"
    points = Points(
        [
            1 + 0j, -1 + 0j,
        ]
    )
