from ..modulation import Modulation

import numpy as np

class FSK(Modulation):
    """Frequency-shift keying parent"""
    def encode(self, data):
        raise NotImplementedError()
    def decode(self, data):
        # see https://wiki.gnuradio.org/index.php/Quadrature_Demod
        data = 0.5 * np.angle(data[0:-1] * np.conj(data[1:]))
        return data
