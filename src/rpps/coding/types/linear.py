import numpy as np

from ._code import _code, register

from ..blocker import block, unblock
from ...errors import DecodeError

@register("linear")
class linear(_code):
    def __init__(self, log, generator, check):
        super().__init__(log, None, generator.shape[0], generator.shape[1])
        self.generator = generator
        self.check = np.transpose(check)
    @classmethod
    def from_config(cls, log, obj):
        from .. import matrix
        if obj.get("p") is None: # Explicit generator and check matrices: (n, k) are their shape
            g, h = np.array(obj["g"], dtype=bool), np.array(obj["h"], dtype=bool)
        else: # Parity block only
            n, k = obj["code"]
            g, h = matrix.linear(n, k, np.array(obj["p"], dtype=bool))
        return cls(log, g, h)

    def encode(self, bits: np.ndarray):
        blocks = block(bits, self.num)
        self.log.trace(f"encoding {len(bits)}, {blocks.shape}")

        encoded = np.empty((len(blocks), self.den), dtype=int)
        for i, blk in enumerate(blocks):
            encoded[i] = np.matmul(blk.astype(int), self.generator)
        encoded = unblock(encoded) % 2
        self.log.trace(f"encoded {len(encoded)}")
        return encoded.astype(bool)

    def decode(self, bits: np.ndarray):
        """Hard decode by syndrome: corrects a single bit error (any more raises DecodeError)"""
        blocks = block(bits, self.den).astype(np.uint8)
        self.log.trace(f"decoding {len(bits)}, {blocks.shape}")
        synd = (blocks.astype(int) @ self.check.astype(int)) % 2 # (blocks, n-k)
        bad = np.nonzero(synd.any(axis=1))[0]
        if len(bad):
            cols = {tuple(row): i for i, row in enumerate(self.check.astype(int) % 2)} # Syndrome of a flip of bit i
            for b in bad:
                i = cols.get(tuple(synd[b]))
                if i is None:
                    raise DecodeError(f"Block {b} has an uncorrectable error pattern")
                blocks[b, i] ^= 1
        return unblock(blocks[:, 0:self.num]).astype(bool)
