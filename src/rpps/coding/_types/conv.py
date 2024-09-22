import numpy as np

from ._code import _code

class conv(_code):
    def __init__(self, num: int, den: int, generator, constraint: int):
        self.generator = generator

        self.num = num
        self.den = den

        self.register = np.zeros((constraint), dtype=bool)

    def encode(self, bits):
        output = np.zeros((self.den,), dtype=bool)

        self.register[0] = bits[0]
        for i, g in enumerate(self.generator):
            output[i] = np.bitwise_xor.reduce(self.register[g])
        #input(f"encode {bits}: {output.astype(int)} // {self.register.astype(int)}")

        self.register = np.roll(self.register, 1)
        return output
