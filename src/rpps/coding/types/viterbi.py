import numpy as np

from ._code import _code

from ..blocker import block, unblock

class viterbi(_code):
    """Viterbi decoder"""
    def __init__(self, num, den, constraint):
        super().__init__(num, den)
        self.con = constraint

    def decode(self, bits: np.ndarray):
        blocks = block(bits, self.den)
        trellis_length = len(blocks)
        trellis_states = 2**self.con

        states = np.arange(trellis_states, dtype=np.uint8)
        states = np.unpackbits(states).astype(bool).reshape(-1,8)[:,-self.con:]

        b_metric = []
        p_metric = []


        print(f"Viterbi decode!")
        print(f"{trellis_length = }")
        print(f"{trellis_states = }")
