import numpy as np

from ._code import _code, register

from ..blocker import block, unblock

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
        blocks = block(bits, self.den)
        self.log.trace(f"decoding {len(bits)}, {blocks.shape}")

        decoded = np.empty((len(blocks), self.den - self.num), dtype=int)

        for i, blk in enumerate(blocks):
            decoded[i] = np.matmul(blk.astype(int), self.check)

        decoded = unblock(decoded)
        self.log.trace(f"decoded {len(decoded)}")
        parity_bits = decoded % 2
        if sum(parity_bits) == 0:
            return unblock(blocks[:,0:self.num])
        print(f"{parity_bits}")
        raise NotImplementedError("Bit error!")
