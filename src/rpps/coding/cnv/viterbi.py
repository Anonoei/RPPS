"""Viterbi convolutional coding"""
from ..coding import Convolutional, Decision

from .. import Meta
from .. import base
from .. import dobject

import numpy as np

class Viterbi(Convolutional):
    """Parent viterbi implementation"""
    name = "Viterbi"

class Hard(Viterbi):
    """Hard decision viterbi"""
    decision = Decision.HARD

    def encode(self, dobj: dobject.BitObject):
        assert self.den - 1 == self.num

        codewords = dobj.data.reshape(-1, self.num)
        print(f"Encoding {codewords}")

        encoded_data = np.zeros((len(codewords), self.den), dtype=bool)

        for idx, codeword in enumerate(codewords):
            if len(codeword) != self.num:
                raise ValueError(f"Data bit vector must be length {self.num}")

            # Check parity
            num_ones = sum(codeword)
            if num_ones % 2 == 0:  # even number of ones
                encoded_data[idx, -1] = False  # parity bit is zero (even)
            else:
                encoded_data[idx, -1] = True  # parity bit is one (odd)

            # Replicate data bits into the first three positions
            encoded_data[idx, 0 : self.num] = codewords[idx]

        print(f"Encoded: {encoded_data.astype(int)}")
        encoded_data = encoded_data.reshape(-1)
        self.log.trace(f"Encoded to {encoded_data}")
        return dobject.CodingData(encoded_data.astype(bool))

    def decode(self, dobj: dobject.BitObject):
        assert self.den - 1 == self.num

        encoded_data = dobj.data.reshape(-1, self.den)
        print(f"Decoding {encoded_data.astype(int)}")

        decoded_data = decoded_data.reshape(-1)
        self.log.trace(f"Decoded to {decoded_data}")
        return dobject.BitObject(decoded_data)


class Soft(Viterbi):
    """Soft decision viterbi"""
    decision = Decision.SOFT
