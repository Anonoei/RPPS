import numpy as np

class _code:
    def __init__(self, num, den):
        self.num = num
        self.den = den
        self.rate = num/den
    def encode(self, bits: np.ndarray):
        """Encode bits"""
        raise NotImplementedError()

    def decode(self, bits: np.ndarray):
        """Decode bits"""
        raise NotImplementedError()
