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

    @property
    def isps(self):
        return int(round(self.sps,0))

class _Stream:
    """Information that follows the stream across loops of processing"""
    __slots__ = ("blocks", "delay", "det")
    def __init__(self):
        self.blocks = 0 # Loops of processing started
        self.delay = 0.0 # Accumulated group delay, in seconds
        self.det = None # Detection result (baud, modulation, timing, cfo...), see rpps.analysis.modulation

class Meta:
    """Signal information shared by every DSP operation

    Three tiers:
        _init/op/state: Fs, sps, cf. Per loop, `reset()` returns them to the source values
        stream: Stream-wide info (group delay, block count). Survives `reset()`
        ctx: Per-stage carried state (filter history, oscillator phase, loop filters...).
            Survives `reset()`. Stage settings stay on the stage, its state lives here.
    `restart()` clears everything for a new stream.
    """
    __slots__ = (
        "_init", "op", "state",
        "ctx", "stream",
        "obj"
    )
    def __init__(self, Fs=1.0, sps=0.0, cf=0.0):
        self._init = _Meta(Fs, sps, cf)
        self.op = _Meta(Fs, sps, cf)
        self.state = _Meta(Fs, sps, cf)

        self.ctx = {}
        self.stream = _Stream()

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
    def isps(self):
        return self.state.isps

    @property
    def cf(self):
        return self.state.cf
    @cf.setter
    def cf(self, cf):
        self.state.cf = cf
        self.op.cf = cf

    def reset(self):
        """Start a new loop of processing. Carried stage state and stream info are kept"""
        self.state.set(self._init.Fs, self._init.sps, self._init.cf)
        self.op.set(self._init.Fs, self._init.sps, self._init.cf)
        self.stream.blocks += 1

    def restart(self):
        """Start a new stream. Drops all carried stage state"""
        self.reset()
        self.ctx.clear()
        self.stream = _Stream()

    def calibrate_sps(self, sps):
        """Declare the sps at the current stage of the chain

        `reset()` restores the source values each loop, so the source sps is derived from the
        current sps through the Fs ratio (decimation/resampling scale both).
        """
        self._init.sps = sps * self._init.Fs / self.state.Fs
        self.state.sps = sps
        self.op.sps = sps

    def state_of(self, stage):
        """Get (creating on first use) the carried state of a stage"""
        try:
            return self.ctx[stage]
        except KeyError:
            st = self.ctx[stage] = stage.new_state(self)
            return st

    @property
    def delay(self):
        """Accumulated group delay in seconds"""
        return self.stream.delay

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
