import numpy as np

class Meta:
    __slots__ = (
        "_init",
        "Fs", "sr", "sps", "cf",
        "obj"
    )
    def __init__(self, Fs, sr=0.0, sps=0.0, cf=0.0):
        self._init = {"Fs": Fs, "sr": sr, "sps": sps, "cf": cf}
        self.Fs = Fs
        self.sr = sr
        self.sps = sps
        self.cf = cf

        self.obj: np.ndarray = None # type: ignore

    def reset(self):
        self.Fs = self._init["Fs"]
        self.sr = self._init["sr"]
        self.sps = self._init["sps"]
        self.cf = self._init["cf"]

    def __str__(self):
        return f"Meta: Fs{self.Fs}, sps{self.sps}"
