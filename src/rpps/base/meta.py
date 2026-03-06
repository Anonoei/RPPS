import numpy as np

class _Meta:
    __slots__ = (
        "Fs", "sps", "cf"
    )
    def __init__(self, Fs, sps, cf):
        self.Fs = Fs
        self.sps = sps
        self.cf = cf

    def set(self, Fs, sps, cf):
        self.Fs = Fs
        self.sps = sps
        self.cf = cf

    @property
    def Ts(self):
        return 1/self.Fs

    @property
    def sr(self):
        return self.Fs/self.sps

class Meta:
    __slots__ = (
        "_init", "op", "state",
        "obj"
    )
    def __init__(self, Fs=1.0, sps=0.0, cf=0.0):
        self._init = _Meta(Fs, sps, cf)
        self.op = _Meta(Fs, sps, cf)
        self.state = _Meta(Fs, sps, cf)

        self.obj: np.ndarray = None # type: ignore

    @property
    def Fs(self):
        return self.state.Fs
    @Fs.setter
    def Fs(self, Fs):
        self.state.Fs = Fs
        self.op.Fs = Fs

    @property
    def sps(self):
        return self.state.sps
    @sps.setter
    def sps(self, sps):
        self.state.sps = sps
        self.op.sps = sps

    @property
    def cf(self):
        return self.state.cf
    @cf.setter
    def cf(self, cf):
        self.state.cf = cf
        self.op.cf = cf

    def reset(self):
        self.state.set(self._init.Fs, self._init.sps, self._init.cf)
        self.op.set(self._init.Fs, self._init.sps, self._init.cf)

    def set(self, Fs=None, sps=None, cf=None):
        Fs = self.Fs if Fs is None else Fs
        sps = self.sps if sps is None else sps
        cf = self.cf if cf is None else cf
        self._init.set(Fs, sps, cf)
        self.state.set(Fs, sps, cf)
        self.op.set(Fs, sps, cf)

    def finish(self):
        self.state.set(self.op.Fs, self.op.sps, self.op.cf)

    @property
    def Ts(self):
        return self.state.Ts

    def __str__(self):
        return f"Meta: Fs{self.state.Fs}, sps{self.state.sps}"

    def str_init(self):
        return f"Fs{self._init.Fs}, sps{self._init.sps}, cf{self._init.cf}"

    def str_op(self):
        return f"Fs{self.op.Fs}, sps{self.op.sps}, cf{self.op.cf}"

    def get_init(self):
        return self._init
