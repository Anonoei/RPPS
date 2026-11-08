import numpy as np

from ._code import _code, register

from ..blocker import block, unblock

TERMINATIONS = ("truncated", "zero", "tailbiting")

def termination_of(obj):
    """A definition's "termination" ("zero" | "tailbiting" | "truncated"), or the older "terminated": true/false"""
    t = obj.get("termination")
    if t is None:
        return "zero" if obj.get("terminated", False) else "truncated"
    if t is True or t is False:
        return "zero" if t else "truncated"
    if t not in TERMINATIONS:
        raise ValueError(f"termination {t!r}: one of {TERMINATIONS}")
    return t

def generators_of(obj):
    """(den, K) generator rows of a definition. Rows of K bits, or numbers ("0o133", 91) with "constraint": the most
    significant of the K bits multiplies the current input"""
    g = obj["generator"]
    K = obj.get("constraint")
    if all(isinstance(x, (int, str)) for x in g):
        if K is None:
            raise ValueError("Generators given as numbers need \"constraint\"")
        nums = [int(x, 0) if isinstance(x, str) else int(x) for x in g]
        for x in nums:
            if x >= 1 << K:
                raise ValueError(f"Generator {x:#o} is longer than the constraint length {K}")
        return np.array([[(x >> (K - 1 - i)) & 1 for i in range(K)] for x in nums], dtype=bool)
    return np.array(g, dtype=bool)

@register("conv")
class conv(_code):
    """Convolutional encoder, rate 1/n. See `viterbi` for the conventions

    puncture: optional (n, P) 0/1 pattern, 1 = send
    termination: "zero" flushes the register with K-1 zeros so the decoder can end in state 0 (the older
    terminated=True); "tailbiting" starts the register with the frame's last K-1 bits, so it ends where it began
    and nothing is added (LTE); "truncated" (default) just stops. A tail-biting code works on whole frames: in a
    stream the bits are collected until the stream is flushed.
    """
    @classmethod
    def from_config(cls, log, obj):
        return cls(log, obj["num"], obj["den"], generators_of(obj), obj["constraint"],
                   obj.get("puncture"), termination_of(obj))

    def __init__(self, log, num: int, den: int, generator, constraint: int, puncture=None, terminated=False):
        super().__init__(log, None, num, den)
        self.generator = np.asarray(generator, dtype=bool).reshape(den, constraint)
        self.con = constraint
        self.puncture = None if puncture is None else np.asarray(puncture, dtype=bool)
        self.termination = termination_of({"termination": terminated})
        self.stream = self.termination != "tailbiting"

    @property
    def terminated(self):
        return self.termination == "zero"

    def encode(self, bits, state=None, final=True):
        """Encode a message. Without `state` it is a frame of its own, starting from the zero register.

        With `state` (a dict, the encoder's part of the stream) the register carries on from the previous call, and
        a terminated code only gets its tail when `final`, which also ends the stream (the state is cleared).
        """
        bits = np.asarray(bits).astype(bool)
        reg = np.zeros(self.con, dtype=bool) if not state or "reg" not in state else state["reg"].copy()
        if self.termination == "tailbiting":
            if state is not None and not final:
                raise ValueError("A tail-biting code encodes whole frames")
            past = bits[::-1][:self.con - 1] # Newest past bit first: the frame's last bits
            reg[1:1 + len(past)] = past
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
