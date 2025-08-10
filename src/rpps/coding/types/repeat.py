import numpy as np

from ._code import _code

from ..blocker import block, unblock

class repeat(_code):
    def __init__(self, count):
        super().__init__(1, count)

    def encode(self, bits: np.ndarray):
        blocks = block(bits, self.num)

        encoded = np.empty((len(blocks), self.den), dtype=bool)

        for i, blk in enumerate(blocks):
            encoded[i,:] = np.repeat(blk, self.den)

        encoded = unblock(encoded).astype(bool)

        assert len(encoded) == len(bits) * self.den
        return encoded

    def decode(self, bits: np.ndarray):
        blocks = block(bits, self.den)

        decoded = np.empty((len(blocks), self.num), dtype=bool)

        codewords = np.array([np.zeros(self.den, dtype=bool), np.ones(self.den, dtype=bool)])

        for i, blk in enumerate(blocks):
            dist = np.empty((2))
            for j, code in enumerate(codewords):
                dist[j] = np.bitwise_xor(code, blk).sum()
            decoded[i] = np.where(dist == dist.min())[0][0]
        decoded = unblock(decoded).astype(bool)
        assert len(bits) // self.den == len(decoded)
        return decoded
