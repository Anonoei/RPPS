import numpy as np

class Convolutional:
    def __init__(self, k, passthrough, polys):
        self.input_size = k
        assert len(passthrough) == polys.shape[0]
        self.passthrough = passthrough
        self.polys = polys
        self.output_size = polys.shape[0]
        self.register = np.zeros((self.output_size), dtype=bool)

    def code(self, bits):
        assert len(bits) == self.input_size

        output = np.zeros(self.output_size, dtype=bool)
        for i, p in enumerate(self.polys):
            if self.passthrough[i]:
                output[i] = bits[i]
                continue
            output[i] = np.bitwise_xor.reduce(self.register[p])

        self.register[0] = bits[0]
        self.register = np.roll(self.register, 1)
        return output
