import numpy as np

from ._code import _code, register
from ...math.gf2 import parity_to_gh

def parity_rows(parity, n, k):
    """The (k, n-k) parity block of parity equations: parity bit j is the XOR of the message bits parity[j] lists"""
    if len(parity) != n - k:
        raise ValueError(f"parity: {len(parity)} equations for {n - k} parity bits")
    p = np.zeros((k, n - k), dtype=np.uint8)
    for j, eq in enumerate(parity):
        for i in eq:
            if not 0 <= int(i) < k:
                raise ValueError(f"parity bit {j}: message bit {i} is not one of 0..{k-1}")
            p[int(i), j] ^= 1
    return p

def extended(G, H):
    """(G, H) with an overall parity bit after the codeword: each generator row gets its weight's parity, and the check
    matrix a row of ones over every bit"""
    G, H = np.asarray(G, dtype=np.uint8), np.asarray(H, dtype=np.uint8)
    G2 = np.concatenate((G, (G.sum(axis=1) % 2)[:, None].astype(np.uint8)), axis=1)
    H2 = np.concatenate((np.concatenate((H, np.zeros((len(H), 1), np.uint8)), axis=1), np.ones((1, G2.shape[1]), np.uint8)))
    return G2, H2


@register("linear")
class linear(_code):
    """Linear block code from its matrices, systematic with the message first. Own decoder: single-error syndrome
    correction (the `trellis` decoder is the default, `syndrome` corrects up to t errors)"""
    t = None # Errors a syndrome decoder should correct; None: from the minimum distance
    extends = True # "extend" goes into its matrices (rpps.code.types.layout wraps the codes that have none)

    def __init__(self, log, generator, check):
        super().__init__(log, None, generator.shape[0], generator.shape[1])
        self.generator = generator
        self.check = np.transpose(check)
    @classmethod
    def from_config(cls, log, obj):
        """One of: "g" + "h" (the matrices), "p" (the k rows of the parity block, n and k) or "parity" (each of the n-k
        parity bits as the message bits it XORs: [[0, 1, 3], ...], the way standards write P0 = I0 + I1 + I3; n and k).
        "extend": an overall parity bit after the codeword"""
        forms = [f for f in ("p", "parity", "g") if obj.get(f) is not None]
        if len(forms) != 1:
            raise ValueError(f"linear code: give one of \"g\" (with \"h\"), \"p\" or \"parity\", not {forms or 'none'}")
        if forms == ["g"]: # Explicit generator and check matrices: (n, k) are their shape
            g, h = np.array(obj["g"], dtype=np.uint8), np.array(obj["h"], dtype=np.uint8)
        else:
            n, k = int(obj["n"]), int(obj["k"])
            p = np.array(obj["p"], dtype=np.uint8) if forms == ["p"] else parity_rows(obj["parity"], n, k)
            g, h = parity_to_gh(n, k, p)
        if obj.get("extend"):
            g, h = extended(g, h)
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
