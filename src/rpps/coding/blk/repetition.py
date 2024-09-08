"""Repetition block coding"""
from ..coding import Block

from .. import dobject

import numpy as np

class Repetition(Block):
    """Repetition coding implementation"""
    name = "Repetition"
    def __init__(self, count):
        super().__init__(1, count, count)

    def encode(self, dobj: dobject.BitObject):
        self.log.trace(f"Encoding {dobj.data}")
        encoded_data = dobject.CodingData()
        for bit in dobj:
            for _ in range(self.length):
                encoded_data.append(bit)
        self.log.trace(f"Encoded to {encoded_data}")
        assert len(dobj) * self.length == len(encoded_data)
        return encoded_data

    def decode(self, dobj: dobject.BitObject):
        decoded_data = dobject.BitObject()
        for i in range(0, len(dobj), self.length):
            bits = dobj[i : i + self.length]
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

        assert len(dobj) // self.length == len(decoded_data)
        return decoded_data
