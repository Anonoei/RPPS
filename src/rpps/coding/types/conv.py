import numpy as np

from ._code import _code, register

from ..blocker import block, unblock

@register("conv")
class conv(_code):
    """Convolutional encoder, rate 1/n. See `viterbi` for the conventions

    puncture: optional (n, P) 0/1 pattern, 1 = send
    terminated: flush the register with K-1 zeros so the decoder can end in state 0
    """
    @classmethod
    def from_config(cls, log, obj):
        return cls(log, obj["num"], obj["den"], np.array(obj["generator"], dtype=bool), obj["constraint"],
                   obj.get("puncture"), obj.get("terminated", False))

    def __init__(self, log, num: int, den: int, generator, constraint: int, puncture=None, terminated=False):
        super().__init__(log, None, num, den)
        self.generator = np.asarray(generator, dtype=bool).reshape(den, constraint)
        self.con = constraint
        self.puncture = None if puncture is None else np.asarray(puncture, dtype=bool)
        self.terminated = terminated

    def encode(self, bits, state=None, final=True):
        """Encode a message. Without `state` it is a frame of its own, starting from the zero register.

        With `state` (a dict, the encoder's part of the stream) the register carries on from the previous call, and
        a terminated code only gets its tail when `final`, which also ends the stream (the state is cleared).
        """
        bits = np.asarray(bits).astype(bool)
        reg = np.zeros(self.con, dtype=bool) if not state or "reg" not in state else state["reg"].copy()
        if self.terminated and final:
            bits = np.concatenate((bits, np.zeros(self.con-1, dtype=bool)))
        blocks = block(bits[:len(bits)//self.num*self.num], self.num)
        self.log.trace(f"encoding {len(bits)}, {blocks.shape}")

        encoded = np.empty((len(blocks), self.den), dtype=int)

        for i, blk in enumerate(blocks):
            reg[0] = blk[0]
            for j, g in enumerate(self.generator):
                encoded[i, j] = np.bitwise_xor.reduce(reg[g])
            reg = np.roll(reg, 1)
            reg[0] = 0

        if state is not None:
            if final:
                state.clear()
            else:
                state["reg"] = reg
        if self.puncture is not None:
            P = self.puncture.shape[1]
            full = len(encoded)//P*P
            mask = np.tile(self.puncture.T, (full//P, 1))
            encoded = encoded[:full][mask]
        encoded = unblock(encoded).astype(bool)
        self.log.trace(f"encoded {len(encoded)}")
        return encoded
