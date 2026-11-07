"""Frequencies as numbers or as people write them"""
import re

_UNITS = {"": 1.0, "h": 1.0, "k": 1e3, "m": 1e6, "g": 1e9}
_PARSE = re.compile(r"^\s*([+-]?(?:\d+\.?\d*|\.\d+)(?:e[+-]?\d+)?)\s*([hkmg]?)(?:hz)?\s*$", re.I)

class Frequency(float):
    """A frequency in Hz. Built from a number or text: `Frequency(99.5e6)`, `Frequency("99.5M")`, `Frequency("2.4 GHz")`, `Frequency("1e6")`

    It is a float, so it goes anywhere a frequency is wanted (`Fs=Frequency("2.4M")`)."""
    def __new__(cls, freq=0.0):
        if isinstance(freq, str):
            m = _PARSE.match(freq)
            if m is None:
                raise ValueError(f"Invalid frequency {freq!r}: use a number with a unit (433.92M, 2.4GHz, 10k) or scientific notation (1e6)")
            freq = float(m.group(1)) * _UNITS[m.group(2).lower()]
        return super().__new__(cls, freq)

    @property
    def raw(self):
        """The frequency in Hz"""
        return float(self)

    @property
    def mult(self):
        """Divisor of the unit it is shown in (1, 1e3, 1e6, 1e9)"""
        for m in (1e9, 1e6, 1e3):
            if abs(self) >= m:
                return m
        return 1.0

    @property
    def short(self):
        """The value in its unit"""
        return float(self)/self.mult

    @property
    def unit(self):
        return {1.0: "", 1e3: "K", 1e6: "M", 1e9: "G"}[self.mult] + "Hz"

    @property
    def str(self):
        return f"{self.short:.2f}{self.unit}"

    def __str__(self):
        return self.str
