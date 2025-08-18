import numpy as np

from ._code import _code

class rs(_code):
    def __init__(self, log, num: int, den: int):
        super().__init__(log, None, num, den)

    def encode(self, bits):
        self.log.trace(f"encoding {len(bits)}")

    def decode(self, bits):
        self.log.trace(f"decoding {len(bits)}")
