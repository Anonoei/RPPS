import numpy as np

from ... import _config

REGISTRY = {}

def register(name):
    """Register a code implementation under the config "type" it is loaded from"""
    def deco(cls):
        REGISTRY[name] = cls
        return cls
    return deco

def word_errors(received_hard, decided, ok=None):
    """Per-codeword decode status: bits that differ between what came in (hard decided) and the codeword the
    decoder chose, or -1 where `ok` is False (the decoder could not find a codeword)"""
    rx = np.asarray(received_hard).astype(np.uint8).reshape(len(decided), -1)
    out = np.count_nonzero(rx != np.asarray(decided).astype(np.uint8), axis=1).astype(int)
    if ok is not None:
        out[~np.asarray(ok, dtype=bool)] = -1
    return out

class _code:
    """A code implementation. Block decoders set `errors` after each decode(): one entry per codeword, the number
    of bits (symbols for Reed-Solomon) corrected, or -1 where the word could not be decoded. Decoding never raises
    on a bad word: its message bits are returned as received and the status says so."""
    soft = False # Does decode() make use of LLR input (otherwise it takes hard decisions)
    errors = None
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
