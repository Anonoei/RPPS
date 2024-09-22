import numpy as np

from ._code import _code

class rs(_code):

    def __init__(self, num: int, den: int):
        self.num = num
        self.den = den

    def encode(self, bits):
        pass

    def decode(self, bits):
        pass
