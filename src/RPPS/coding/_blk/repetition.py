from ..block import Block
from helpers.stream import Stream
from helpers.bitarray import bitarray

import numpy as np

class Repetition(Block):
    def __init__(self, num, den):
        super().__init__(num, den, den)

    def encode(self, data: Stream):
        encoded_data = bitarray()
        for bit in data.bitarray:
            for _ in range(self.length):
                encoded_data.append(bit)
        return encoded_data

    def decode(self, encoded_data: bitarray):
        decoded_data = bitarray()
        for i in range(0, len(encoded_data), self.length):
            bits = encoded_data[i:i+self.length]
            diff = self.length - sum(bits)
            if self.length/2 > diff:
                decoded_data.append(1)
            else:
                decoded_data.append(0)
        return decoded_data
