import numpy as np

from ._code import _code, err

from ..blocker import block, unblock

class repeat(_code):
    def __init__(self, log, count):
        super().__init__(log, f"{count}", 1, count)

    def encode(self, bits: np.ndarray):
        self.log.trace(f"encoding {len(bits)}")
        encoded = np.repeat(bits, self.den)
        self.log.trace(f"encoded {len(encoded)}")
        return encoded

    def decode(self, bits: np.ndarray):
        blocks = block(bits, self.den)
        self.log.trace(f"decoding {len(bits)}, {blocks.shape}")

        decoded = np.sum(blocks, axis=1)/self.den
        decoded = np.round(decoded).astype(bits.dtype)
        self.log.trace(f"decoded {len(decoded)}")
        return decoded
