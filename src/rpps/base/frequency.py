"""Float/Int to Freq converter"""
import re

class Frequency:
    """Helper for frequencies"""

    def __init__(self, freq=0):
        self.freq = freq
        self.mult = 1
        self.label = ""

        if isinstance(freq, str):
            if not re.match(r"((\d+\.?)?\d+?)((e\d+)|([hkmg]))", freq):
                print("Invalid frequency string!")
                print("It must be provided in scientific notation")
                print(" ##.#e# - re: '((\\d+\\.?)?\\d+?)(e\\d+)'")
                print("Or with units")
                print(" ##.#m - re: '((\\d+\\.?)?\\d+?)[hkmg]'")
                exit(1)

    def __str__(self):
        return self.str

    def default(self):
        return self.str

    @property
    def raw(self):
        """Return the raw freq value"""
        return self.freq

    @property
    def short(self):
        """Return the shorthand freq"""
        return self.freq / self.mult

    @property
    def str(self):
        """Return the shorthand freq and unit"""
        return format(self.freq / self.mult, ".2f") + self.unit

    @property
    def unit(self):
        """Return the frequency unit"""
        return self.label.upper() + "Hz"

    def init(self, freq=None):
        """Initialize the frequency object"""
        if freq is not None:
            self.freq = freq

        # input("Initializing: " + str(self.freq))
        if isinstance(self.freq, str):
            if "e" in self.freq:
                freq = self.freq.split("e")
                mult = int("1" + (int(freq[1]) * "0"))
                freq = float(freq[0])
                # input("scinot: got " + str(freq) + " * " + str(mult))
                self.freq = freq * mult
                self.init()
                return

            for i, l in enumerate(["h", "k", "m", "g"]):
                if self.freq.endswith(l):
                    self.mult = int("1" + (i * "000"))
                    self.freq = float(self.freq[:-1])
                    self.label = l
                    return
        else:
            for i, l in enumerate(["", "k", "m", "g"]):
                mult = int("1" + (i * "000"))
                cur_freq = self.freq / mult
                if cur_freq < 1000:
                    self.mult = mult
                    self.label = l
                    return
