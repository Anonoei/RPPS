"""Zadoff-Chu sequences: constant amplitude, zero periodic autocorrelation (CAZAC)

    rp.sequence.zadoff_chu(25, 63)          root 25 of length 63 (LTE's PSS punctures its middle element)
    rp.sequence.zadoff_chu(129, 839, 13)    a PRACH root, cyclically shifted by 13

x_u[n] = exp(-j pi u n (n + N mod 2) / N), u coprime to N. Every element has magnitude 1 and the periodic autocorrelation
is zero at every lag but 0, so a received period correlated with the root gives one peak (the channel) and noise alone
at every other lag: rp.sync.quality.from_sequence reads the SNR from that. Two roots of a prime length cross-correlate
at 1/sqrt(N) at every lag.

As a definition (a preamble a profile's `sync frame` step names): {"base": "sequence", "type": "zc", "root": 25,
"length": 63, "shift": 0}; loaded, `.symbols` is the sequence and `.fields` its root and shift.
"""
from math import gcd

import numpy as np

def zadoff_chu(u, N, shift=0):
    """Root u of length N (u coprime to N, 0 < u < N), cyclically advanced by `shift`: x[(n + shift) mod N]"""
    u, N = int(u), int(N)
    if N < 1 or not 0 < u < N or gcd(u, N) != 1:
        raise ValueError(f"Zadoff-Chu root {u} of length {N}: the root must be in 1..N-1 and coprime to N")
    n = (np.arange(N) + int(shift)) % N
    return np.exp(-1j*np.pi*u*n*(n + N % 2)/N)

class ZadoffChu:
    """A Zadoff-Chu sequence definition: `symbols` (complex, unit magnitude), `fields` ({"root", "shift"})"""
    def __init__(self, name, root, length, shift=0):
        self.name, self.root, self.length, self.shift = name, int(root), int(length), int(shift)
        self.symbols = zadoff_chu(self.root, self.length, self.shift)
        self.fields = {"root": self.root, "shift": self.shift}

    def __len__(self):
        return self.length

    def __repr__(self):
        return f"<ZadoffChu {self.name}: root {self.root} of {self.length}, shift {self.shift}>"

    @staticmethod
    def load(name, obj):
        from ..errors import ConfigError
        for k in ("root", "length"):
            if k not in obj:
                raise ConfigError(f"sequence {name}: a zc sequence needs {k!r}")
        try:
            return ZadoffChu(name, obj["root"], obj["length"], obj.get("shift", 0))
        except ValueError as e:
            raise ConfigError(f"sequence {name}: {e}") from None
