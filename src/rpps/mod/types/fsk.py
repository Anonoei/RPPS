from types import SimpleNamespace

from ..modulation import Modulation
from ...base import Stage

import numpy as np

class FSK(Modulation):
    """Frequency-shift keying parent"""
    def encode(self, data):
        raise NotImplementedError()
    def decode(self, data):
        # see https://wiki.gnuradio.org/index.php/Quadrature_Demod
        data = 0.5 * np.angle(data[0:-1] * np.conj(data[1:]))
        return data


class FMDemod(Stage):
    """Quadrature FM demodulator as a stream stage: the last sample carries across blocks"""
    __slots__ = ("gain",)
    def __init__(self, gain=0.5):
        self.gain = gain

    def new_state(self, meta):
        return SimpleNamespace(last=None)

    def step(self, samples, st, meta):
        if len(samples) == 0:
            return np.zeros(0, dtype=np.float32)
        last = samples[0] if st.last is None else st.last
        st.last = samples[-1]
        z = np.concatenate(([last], samples))
        return self.gain * np.angle(z[:-1] * np.conj(z[1:]))
