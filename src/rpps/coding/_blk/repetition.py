from ..coding import Block
from .. import Stream
from .. import bitarray

import numpy as np

class Repetition(Block):
    def __init__(self, count):
        super().__init__(1, count, count)

    def encode(self, data: Stream):
        self.log.trace(f"Encoding {data.bitarray}")
        encoded_data = bitarray()
        for bit in data.bitarray:
            for _ in range(self.length):
                encoded_data.append(bit)
        self.log.trace(f"Encoded to {encoded_data}")
        return encoded_data

    def decode(self, data: bitarray):
        decoded_data = bitarray()
        for i in range(0, len(data), self.length):
            bits = data[i : i + self.length]
            bit_sum = sum(bits)
            if not bit_sum in (0, self.length):
                self.log.trace(f"Correcting bits {bits} - {bit_sum}")
                diff = self.length - bit_sum
                if self.length/2 > diff:
                    bit = 1
                else:
                    bit = 0
            else:
                if bit_sum == 0:
                    bit = 0
                else:
                    bit = 1
            decoded_data.append(bit)
        self.log.trace(f"Decoded to {decoded_data}")
        return decoded_data
