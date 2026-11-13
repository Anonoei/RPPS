"""Chip codes: a set of M chip sequences of N chips each, as +/-1 floats (spreading codes)

    chips   {"base": "sequence", "type": "chips", "chips": [1, -1, ...]}      one sequence, written out (Barker)
            {..., "bits": "10110111000", "one": -1}      or as bits; `one` is the chip a 1 bit becomes (default -1: 0 -> +1)
    set     {"base": "sequence", "type": "set", "sequences": ["1101...", ...], "one": 1}   one per symbol value (802.15.4)
    gold    {"base": "sequence", "type": "gold", "length": 1023, "sequences": [component, component], ...}
            register sequences XOR'd (GPS C/A); a component is the recurrence of rpps.sequence.recurrence
            ({"length": 10, "taps": [0, 7], "init": 1023}).
            "select": the component whose output is read at a delay chosen by a parameter, so one definition covers
            every code of the family: {"component": 1, "stages": {"param": "prn"}, "table": {"1": [2, 6], ...}}
            XORs the register stages the table lists for that value (stage k, 1 = first, is the output k-1 chips ahead of
            the last stage, which is the output itself).
    Any of them may carry "rate": the chip rate in chips/s, and "period": chips per code period (default the length).
"""
import numpy as np

from ..errors import ConfigError
from .recurrence import _Component

class Code:
    """A set of spreading sequences. `chips` is (count, length) of +/-1; `rate` the chip rate in chips/s if the definition gives one"""
    def __init__(self, name, chips, rate=None):
        chips = np.atleast_2d(np.asarray(chips, dtype=np.float64))
        if chips.ndim != 2 or chips.shape[1] == 0 or not np.all(np.abs(chips) == 1):
            raise ValueError("chips must be rows of +1/-1")
        self.name = name
        self.chips = chips
        self.rate = None if rate is None else float(rate)

    @property
    def count(self):
        """Sequences in the set (1 for a single code)"""
        return self.chips.shape[0]

    @property
    def length(self):
        """Chips per sequence"""
        return self.chips.shape[1]

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        """Sequence i (a 1-D array)"""
        return self.chips[i]

    def bits(self, one=-1):
        """The chips as 0/1 (`one`: the chip value that is a 1)"""
        return (self.chips == one).astype(np.uint8)

    def spread(self, symbols, index=0):
        """Each symbol times the sequence: (n,) -> (n*length,). index: which sequence (an int, or one per symbol for a set)"""
        symbols = np.asarray(symbols)
        idx = np.broadcast_to(np.asarray(index, dtype=int), symbols.shape)
        return (symbols[:, None] * self.chips[idx]).reshape(-1)

    def despread(self, chips, index=0):
        """Correlate each length-N block with the sequence, normalised by N: (n*length,) -> (n,)"""
        x = np.asarray(chips).reshape(-1, self.length)
        return x @ self.chips[index] / self.length

    def autocorrelation(self, index=0):
        """Periodic autocorrelation of a sequence (integer lags 0..length-1)"""
        c = self.chips[index]
        f = np.fft.fft(c)
        return np.rint(np.fft.ifft(f * np.conj(f)).real).astype(int)

    def crosscorrelation(self, a, b):
        """Periodic cross-correlation of sequences a and b"""
        fa, fb = np.fft.fft(self.chips[a]), np.fft.fft(self.chips[b])
        return np.rint(np.fft.ifft(fa * np.conj(fb)).real).astype(int)

    def __repr__(self):
        return f"<Code {self.name}: {self.count} x {self.length} chips>"

def _to_chips(bits, one):
    return np.where(np.asarray(bits) == 1, float(one), -float(one))

def _chip_str(s, one):
    s = "".join(str(s).split())
    if set(s) - {"0", "1"}:
        raise ConfigError(f"chip string {s!r}: only 0 and 1")
    return _to_chips([int(c) for c in s], one)

def _fixed(name, obj):
    one = int(obj.get("one", -1))
    if "chips" in obj:
        chips = np.asarray(obj["chips"], dtype=np.float64)
    elif "bits" in obj:
        chips = _chip_str(obj["bits"], one)
    else:
        raise ConfigError(f"{name}: a chips code needs 'chips' or 'bits'")
    return Code(name, chips, obj.get("rate"))

def _pnset(name, obj):
    one = int(obj.get("one", -1))
    rows = [_chip_str(s, one) for s in obj["sequences"]]
    if len({len(r) for r in rows}) != 1:
        raise ConfigError(f"{name}: sequences of different lengths")
    return Code(name, np.array(rows), obj.get("rate"))

def _gold(name, obj):
    n = int(obj["length"])
    one = int(obj.get("one", -1))
    comps = [_Component(c) for c in obj["sequences"]]
    sel = obj.get("select")
    bits = np.zeros(n, dtype=np.uint8)
    for k, c in enumerate(comps):
        offsets = [0]
        if sel and int(sel["component"]) == k:
            key = str(sel["stages"])
            table = sel["table"]
            if key not in table:
                raise ConfigError(f"{name}: {key!r} is not in the table, have {sorted(table, key=lambda s: int(s) if s.isdigit() else s)}")
            offsets = [c.L - int(s) for s in table[key]]
        for o in offsets:
            if not 0 <= o < c.L:
                raise ConfigError(f"{name}: stage out of range for a {c.L}-stage register")
            bits ^= c.first(n + o)[o:o + n]
    return Code(name, _to_chips(bits, one), obj.get("rate"))

LOADERS = {"chips": _fixed, "set": _pnset, "gold": _gold}
