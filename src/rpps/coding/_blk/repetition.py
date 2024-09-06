from ..coding import Block

from .. import Meta
from .. import base
from .. import dobject

import numpy as np

class Repetition(Block):
    name = "Repetition"
    def __init__(self, count):
        super().__init__(1, count, count)

    def encode(self, dobj: dobject.DataObject, meta: Meta):
        self.log.trace(f"Encoding {dobj.stream.bitarray}")
        encoded_data = base.bitarray()
        for bit in dobj.stream.bitarray:
            for _ in range(self.length):
                encoded_data.append(bit)
        self.log.trace(f"Encoded to {encoded_data}")
        retr = dobject.CodingData()
        retr.from_bitarray(encoded_data)
        return retr

    def decode(self, dobj: dobject.DataObject, meta: Meta):
        decoded_data = base.bitarray()
        for i in range(0, len(dobj.stream.bitarray), self.length):
            bits = dobj.stream.bitarray[i : i + self.length]
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
        retr = dobject.CodingData()
        retr.from_bitarray(decoded_data)
        return retr
