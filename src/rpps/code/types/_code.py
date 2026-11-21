import numpy as np

from ... import _config

REGISTRY = {}

def register(name):
    """Register a code implementation under the config "type" it is loaded from"""
    def deco(cls):
        REGISTRY[name] = cls
        return cls
    return deco

def as_llr(x, soft=None, hard=1.0):
    """Decoder input as float64 LLRs (positive means 0). soft: is x LLRs (None: when it is floating point);
    otherwise x is hard bits, which become +-`hard`"""
    x = np.asarray(x)
    if soft is None:
        soft = x.dtype.kind == "f"
    return x.astype(np.float64) if soft else hard*(1.0 - 2.0*x.astype(np.float64))

def word_errors(received_hard, decided, ok=None):
    """Per-codeword decode status: bits that differ between what came in (hard decided) and the codeword the
    decoder chose, or -1 where `ok` is False (the decoder could not find a codeword)"""
    rx = np.asarray(received_hard).astype(np.uint8).reshape(len(decided), -1)
    out = np.count_nonzero(rx != np.asarray(decided).astype(np.uint8), axis=1).astype(int)
    if ok is not None:
        out[~np.asarray(ok, dtype=bool)] = -1
    return out

class _code:
    """A code implementation. `decode(x, status=True)` returns the decode's status with its output: per codeword, the
    number of bits (symbols for Reed-Solomon) corrected, or -1 where the word could not be decoded; None for a decoder
    without codewords (Viterbi, BCJR, an interleaver). Nothing of a decode is kept on the code, so one loaded code can
    decode on several threads. Decoding never raises on a bad word: its message bits are returned as received and the
    status says so.

    Implementations define `_decode(x, ...) -> (output, errors)`; `decode` passes every other argument on."""
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

    def decode(self, bits: np.ndarray, *args, status=False, **kw):
        """Decode bits: the message, or (message, errors per codeword) with status=True"""
        out, errors = self._decode(bits, *args, **kw)
        return (out, errors) if status else out

    def _decode(self, bits: np.ndarray):
        """(output, errors per codeword or None)"""
        raise NotImplementedError()

    def pad(self, bits):
        padded = 0
        while not len(bits) % self.den == 0:
            bits = np.append(bits, 0)
            padded += 1

        if not padded == 0:
            self.log.warn(f"Padded by {padded}, {len(bits)-padded} % {self.den}")
        return bits
