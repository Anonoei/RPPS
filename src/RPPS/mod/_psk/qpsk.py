from ..psk import PSK, Constellation, Logger, np

Logger().Modulation.PSK.Child("QPSK")

class QPSK(PSK):
    constellation = Constellation(
        [
            .7 -.7j, -.7 - .7j,
            .7 + .7j, -.7 + .7j
        ],
        [
            0, 2,
            1, 3,
        ]
    )
