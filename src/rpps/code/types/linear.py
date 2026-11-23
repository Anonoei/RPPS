import numpy as np

from ._code import _code, register


@register("linear")
class linear(_code):
    """Linear block code from its matrices, systematic with the message first. Own decoder: single-error syndrome
    correction (the `trellis` decoder is the default, `syndrome` corrects up to t errors)"""
    t = None # Errors a syndrome decoder should correct; None: from the minimum distance

    def __init__(self, log, generator, check):
        super().__init__(log, None, generator.shape[0], generator.shape[1])
        self.generator = generator
        self.check = np.transpose(check)
    @classmethod
    def from_config(cls, log, obj):
        from ...math.gf2 import parity_to_gh
        if obj.get("p") is None: # Explicit generator and check matrices: (n, k) are their shape
            g, h = np.array(obj["g"], dtype=bool), np.array(obj["h"], dtype=bool)
        else: # Parity block only: "n", "k" and the k rows of "p"
            g, h = parity_to_gh(obj["n"], obj["k"], np.array(obj["p"], dtype=bool))
        code = cls(log, g, h)
        code.t = obj.get("t") # errors a syndrome decoder corrects: needed where the minimum distance is too costly to find
        return code

    def encode(self, bits: np.ndarray):
        blocks = bits.reshape(-1, self.num)
        self.log.trace(f"encoding {len(bits)}, {blocks.shape}")

        encoded = np.empty((len(blocks), self.den), dtype=int)
        for i, blk in enumerate(blocks):
            encoded[i] = np.matmul(blk.astype(int), self.generator)
        encoded = encoded.reshape(-1) % 2
        self.log.trace(f"encoded {len(encoded)}")
        return encoded.astype(bool)

    def _decode(self, bits: np.ndarray):
        """Hard decode by syndrome: corrects a single bit error; a word with any other syndrome is returned as
        received with `errors` -1"""
        blocks = bits.reshape(-1, self.den).astype(np.uint8)
        self.log.trace(f"decoding {len(bits)}, {blocks.shape}")
        synd = (blocks.astype(int) @ self.check.astype(int)) % 2 # (blocks, n-k)
        errors = np.zeros(len(blocks), dtype=int)
        bad = np.nonzero(synd.any(axis=1))[0]
        if len(bad):
            cols = {tuple(row): i for i, row in enumerate(self.check.astype(int) % 2)} # Syndrome of a flip of bit i
            for b in bad:
                i = cols.get(tuple(synd[b]))
                if i is None:
                    errors[b] = -1
                    continue
                blocks[b, i] ^= 1
                errors[b] = 1
        return blocks[:, 0:self.num].reshape(-1).astype(bool), errors
