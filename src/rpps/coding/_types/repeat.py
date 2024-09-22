from .. import dobject

class repeat:
    def __init__(self, count):
        self.num = 1
        self.den = count

    def encode(self, dobj: dobject.BitObject):
        encoded_data = dobject.CodingData()
        for bit in dobj:
            for _ in range(self.den):
                encoded_data.append(bit)
        assert len(dobj) * self.den == len(encoded_data)
        return encoded_data

    def decode(self, dobj: dobject.BitObject):
        decoded_data = dobject.BitObject()
        for i in range(0, len(dobj), self.den):
            bits = dobj[i : i + self.den]
            bit_sum = sum(bits)
            if not bit_sum in (0, self.den):
                diff = self.den - bit_sum
                if self.den / 2 > diff:
                    bit = 1
                else:
                    bit = 0
            else:
                if bit_sum == 0:
                    bit = 0
                else:
                    bit = 1
            decoded_data.append(bit)

        assert len(dobj) // self.den == len(decoded_data)
        return decoded_data
