"""Additive scrambling by a binary sequence defined the way standards define it: a linear recurrence or a shift
register, optionally XOR'd with a second one (a Gold sequence), from a seed that is often a load-time parameter

    {"base": "adt", "type": "sequence",
     "sequences": [component, ...],     XOR'd together (two for a Gold sequence)
     "skip": 1600,                      outputs dropped before the first one used (LTE's Nc)
     "period": 4320}                    optional: restart from the seed every `period` bits (one frame)

A component is a recurrence s[t+L] = XOR of s[t+j] over `taps`, output s[t]:
    {"length": 31, "taps": [0, 3], "init": 1, "init_order": "lsb"}
        init: the first L outputs, as an integer (lsb: bit i is s[i]; msb: bit L-1-i is s[i]) or a list of bits
or a register as drawn (Galois form, as Bluetooth draws its whitening LFSR):
    {"register": "galois", "length": 7, "taps": [4], "state": [1, 0, 1, 0, 0, 1, 1]}
        positions 0..L-1; each step outputs position L-1, shifts toward it, feeds the output into position 0
        and XORs it into each tap position
"""
import numpy as np

from .scrambler import Scram

def _bits(v, L, order="lsb"):
    if isinstance(v, str):
        v = [int(c) for c in v]
    if isinstance(v, (list, tuple, np.ndarray)):
        b = np.asarray(v, dtype=np.uint8)
        if len(b) != L:
            raise ValueError(f"{len(b)} seed bits for a {L}-bit sequence")
        return b
    v = int(v)
    if v >> L:
        raise ValueError(f"seed {v:#x} is wider than {L} bits")
    idx = np.arange(L) if order == "lsb" else np.arange(L - 1, -1, -1)
    return ((v >> idx) & 1).astype(np.uint8)

def _galois_outputs(L, taps, state, n):
    s = list(int(b) for b in state)
    out = []
    for _ in range(n):
        o = s[L - 1]
        out.append(o)
        s = [o] + s[:L - 1]
        for t in taps:
            s[t] ^= o
    return np.array(out, dtype=np.uint8)

def _recurrence_of(seq):
    """Shortest recurrence of a binary sequence (Berlekamp-Massey over GF(2)): (L, taps) with s[t+L] = XOR s[t+j]"""
    s = [int(x) for x in seq]
    c, b = [1], [1]
    L, m = 0, 1
    for n in range(len(s)):
        d = s[n]
        for i in range(1, L + 1):
            d ^= c[i] & s[n - i]
        if d == 0:
            m += 1
            continue
        t = c[:]
        c = c + [0]*(len(b) + m - len(c))
        for i, bi in enumerate(b):
            c[i + m] ^= bi
        if 2*L <= n:
            L, b, m = n + 1 - L, t, 1
        else:
            m += 1
    c = c + [0]*(L + 1 - len(c))
    return L, [L - i for i in range(1, L + 1) if c[i]] # s[t] = XOR c_i s[t-i]  ->  s[t+L] = XOR s[t+L-i]

class _Component:
    """One binary sequence, extended on demand"""
    def __init__(self, obj):
        L = int(obj["length"])
        taps = [int(t) for t in obj["taps"]]
        if obj.get("register") == "galois":
            first = _galois_outputs(L, taps, _bits(obj["state"], L, "msb"), 2*L)
            L2, rtaps = _recurrence_of(first)
            self.L, self.taps, self.s = L2, rtaps, first[:max(L2, 1)].copy()
        elif obj.get("register") not in (None, "recurrence"):
            raise ValueError(f"register {obj['register']!r}: 'galois' or 'recurrence'")
        else:
            if not taps or max(taps) >= L or min(taps) < 0:
                raise ValueError(f"recurrence taps {taps} must lie in 0..{L-1}")
            self.L, self.taps, self.s = L, taps, _bits(obj["init"], L, obj.get("init_order", "lsb"))
        self.step = max(1, self.L - max(self.taps)) if self.taps else self.L

    def first(self, n):
        """s[0 .. n-1]"""
        if len(self.s) < n:
            L, taps, step = self.L, self.taps, self.step
            s = np.concatenate((self.s, np.zeros(n - len(self.s) + step, dtype=np.uint8)))
            t = len(self.s) - L
            while t + L < n:
                c = min(step, n - (t + L))
                v = np.zeros(c, dtype=np.uint8)
                for j in taps:
                    v ^= s[t + j:t + j + c]
                s[t + L:t + L + c] = v
                t += c
            self.s = s[:n]
        return self.s[:n]

class Sequence(Scram):
    """Additive scrambler by a sequence (see the module docstring). encode/decode: data XOR the sequence from
    position `start` (0 = the first bit after `skip`); soft(llr, start) flips the LLRs instead"""
    def __init__(self, components, skip=0, period=None):
        super().__init__()
        self.components = [_Component(c) for c in components]
        if not self.components:
            raise ValueError("A sequence scrambler needs at least one sequence")
        self.skip = int(skip)
        self.period = None if period is None else int(period)

    def sequence(self, n, start=0):
        """n bits of the sequence from position `start`"""
        if self.period:
            idx = (start + np.arange(n)) % self.period
            full = self._raw(self.period)
            return full[idx]
        return self._raw(start + n)[start:]

    def _raw(self, n):
        out = np.zeros(n, dtype=np.uint8)
        for c in self.components:
            out ^= c.first(self.skip + n)[self.skip:]
        return out

    def encode(self, data, start=0):
        bits = np.asarray(data).astype(np.uint8)
        return (bits ^ self.sequence(len(bits), start)).astype(bool)

    decode = encode

    def soft(self, llr, start=0):
        """Descramble LLRs (positive = 0): the sign flips where the sequence is 1"""
        llr = np.asarray(llr, dtype=np.float64)
        return llr*(1.0 - 2.0*self.sequence(len(llr), start))

    def _stream(self, ctx, key, bits, enc):
        start = ctx.get(key, 0)
        out = self.encode(bits, start)
        end = start + len(bits)
        ctx[key] = end % self.period if self.period else end
        return out

    @staticmethod
    def load(name, obj):
        return type(name, (Sequence,), dict(name=name))(obj["sequences"], obj.get("skip", 0), obj.get("period"))
