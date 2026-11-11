import numpy as np

from ._code import _code, register

from ..blocker import block, unblock

@register("repeat")
class repeat(_code):
    soft = True

    def __init__(self, log, count):
        super().__init__(log, f"{count}", 1, count)

    @classmethod
    def from_config(cls, log, obj):
        return cls(log, obj["count"])

    def encode(self, bits: np.ndarray):
        self.log.trace(f"encoding {len(bits)}")
        encoded = np.repeat(bits, self.den)
        self.log.trace(f"encoded {len(encoded)}")
        return encoded

    def decode(self, bits: np.ndarray):
        bits = np.asarray(bits)
        blocks = block(bits[:len(bits)//self.den*self.den], self.den)
        self.log.trace(f"decoding {len(bits)}, {blocks.shape}")

        if bits.dtype.kind == "f": # LLRs add up: the sum is the combined LLR
            decoded = np.sum(blocks, axis=1) < 0
            self.errors = np.count_nonzero((blocks < 0) != decoded[:, None], axis=1)
            return decoded
        ones = np.sum(blocks.astype(int), axis=1)
        decoded = (2*ones > self.den).astype(bits.dtype)
        self.errors = np.minimum(ones, self.den - ones)
        self.errors[2*ones == self.den] = -1 # A tie (even count): no majority
        self.log.trace(f"decoded {len(decoded)}")
        return decoded
