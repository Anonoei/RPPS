import numpy as np

from ... import _config

REGISTRY = {}

def register(name):
    """Register a code implementation under the config "type" it is loaded from"""
    def deco(cls):
        REGISTRY[name] = cls
        return cls
    return deco

class _code:
    soft = False # Does decode() make use of LLR input (otherwise it takes hard decisions)
    def __init__(self, log, name, num, den):
        if name is None:
            name = f"{num}_{den}"
        self.log = log.Child(f"{name}", _config.LOG_CODING_TYPES)
        self.num = num
        self.den = den
        self.rate = num/den
    def encode(self, bits: np.ndarray):
        """Encode bits"""
        raise NotImplementedError()

    def decode(self, bits: np.ndarray):
        """Decode bits"""
        raise NotImplementedError()

    def pad(self, bits):
        padded = 0
        while not len(bits) % self.den == 0:
            bits = np.append(bits, 0)
            padded += 1

        if not padded == 0:
            self.log.warn(f"Padded by {padded}, {len(bits)-padded} % {self.den}")
        return bits
