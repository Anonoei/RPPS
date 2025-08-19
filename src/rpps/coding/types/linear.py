import numpy as np

from ._code import _code

from ..blocker import block, unblock

class linear(_code):
    def __init__(self, log, generator, check):
        super().__init__(log, None, generator.shape[0], generator.shape[1])
        self.generator = generator
        self.check = np.transpose(check)
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
