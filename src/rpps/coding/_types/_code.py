import numpy as np

class _code:
    def encode(self, bits: np.ndarray):
        """Encode bits"""
        raise NotImplementedError()

    def decode(self, bits: np.ndarray):
        """Decode bits"""
        raise NotImplementedError()
