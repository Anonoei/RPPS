import numpy as np

from .. import dobject

class repeat:
    def __init__(self, count):
        self.count = count

    def encode(self, dobj: dobject.BitObject):
        encoded_data = dobject.CodingData()
        for bit in dobj:
            for _ in range(self.count):
                encoded_data.append(bit)
        assert len(dobj) * self.count == len(encoded_data)
        return encoded_data

    def decode(self, dobj: dobject.BitObject):
        decoded_data = dobject.BitObject()
        for i in range(0, len(dobj), self.count):
            bits = dobj[i : i + self.count]
            bit_sum = sum(bits)
            if not bit_sum in (0, self.count):
                diff = self.count - bit_sum
                if self.count / 2 > diff:
                    bit = 1
                else:
                    bit = 0
            else:
                if bit_sum == 0:
                    bit = 0
                else:
                    bit = 1
            decoded_data.append(bit)

        assert len(dobj) // self.count == len(decoded_data)
        return decoded_data
