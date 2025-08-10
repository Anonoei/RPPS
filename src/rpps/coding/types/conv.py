import numpy as np

from ._code import _code

from ..blocker import block, unblock

class conv(_code):
    def __init__(self, num: int, den: int, generator, constraint: int):
        super().__init__(num, den)
        self.generator = generator

        self.register = np.zeros((constraint), dtype=bool)

    def encode(self, bits):
        blocks = block(bits, self.num)

        encoded = np.empty((len(blocks), self.den), dtype=int)

        for i, blk in enumerate(blocks):
            self.register[0] = blk[0]
            for j, g in enumerate(self.generator):
                encoded[i, j] = np.bitwise_xor.reduce(self.register[g])
            self.register = np.roll(self.register, 1)
            self.register[0] = 0

            # input(f"encode {blk}: {encoded[i].astype(int)} // {self.register.astype(int)}")

        encoded = unblock(encoded).astype(bool)

        return encoded
