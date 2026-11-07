"""Rate information and stream state that travel with data

Users don't build these: give `rp.Data` an Fs / sps / cf and the stages take care of the rest.
"""

class Rate:
    """Sample rate (Fs), samples per symbol (sps) and centre frequency (cf)"""
    __slots__ = ("Fs", "sps", "cf")
    def __init__(self, Fs, sps, cf):
        self.Fs = Fs
        self.sps = sps
        self.cf = cf

    def set(self, Fs, sps, cf):
        self.Fs = Fs
        self.sps = sps
        self.cf = cf

    def copy(self):
        return Rate(self.Fs, self.sps, self.cf)

    @property
    def Ts(self):
        return 1/self.Fs

    @property
    def sr(self):
        return self.Fs/self.sps

    @property
    def isps(self):
        return int(round(self.sps, 0))

    def __repr__(self):
        return f"Rate(Fs={self.Fs}, sps={self.sps}, cf={self.cf})"

class Stream:
    """Everything that follows one source across its blocks, shared by every block (and every data derived from it)

    states: the carried state of each stage (filter history, oscillator phase, loop filters...), keyed by the stage,
        so a stage object is reusable and the same stage in two streams has two states
    det: detection result (baud, modulation, timing, cfo...), see rpps.analysis.modulation
    """
    __slots__ = ("states", "det", "blocks", "src_Fs", "src_sps")
    def __init__(self, Fs=None):
        self.states = {}
        self.det = None
        self.blocks = 0 # Blocks the source has produced
        self.src_Fs = Fs # Rate of the source
        self.src_sps = None # sps at the source, once a stage has learned it (Acquire): the next blocks start from it

    def restart(self):
        """Forget everything: the next block starts a new stream"""
        self.states.clear()
        self.det = None
        self.blocks = 0
        self.src_sps = None

class Meta:
    """The rate information of one block of data, as a stage sees it

        Fs, sps, cf, isps, Ts: the rates of the block coming in (read-only for stages)
        op: the rates of the block going out. A stage changes these (meta.op.Fs /= 2), the carrier applies them
        delay: group delay of the data so far [s]. A stage adds its own
        stream: the Stream this block belongs to (stage state, detection)
    """
    __slots__ = ("state", "op", "delay", "stream")
    def __init__(self, Fs=1.0, sps=0.0, cf=0.0, delay=0.0, stream=None):
        self.stream = Stream(Fs) if stream is None else stream
        if self.stream.src_Fs is None:
            self.stream.src_Fs = Fs
        if self.stream.src_sps is not None:
            sps = self.stream.src_sps # A stage already learned the source's sps
        self.state = Rate(Fs, sps, cf)
        self.op = Rate(Fs, sps, cf)
        self.delay = delay

    @property
    def Fs(self):
        return self.state.Fs
    @property
    def sps(self):
        return self.state.sps
    @property
    def cf(self):
        return self.state.cf
    @property
    def isps(self):
        return self.state.isps
    @property
    def Ts(self):
        return self.state.Ts
    @property
    def sr(self):
        return self.state.sr

    def fork(self):
        """Meta for the data a stage produces: the output rates become the input rates. Stream is shared"""
        m = Meta.__new__(Meta)
        m.stream = self.stream
        m.state = self.op.copy()
        m.op = self.op.copy()
        m.delay = self.delay
        return m

    def calibrate_sps(self, sps):
        """Declare the sps at the current stage of the chain

        Blocks come from the source with the source's sps, so the sps at the source is worked out through the
        Fs ratio (decimation/resampling scale both) and handed to the blocks that follow.
        """
        self.stream.src_sps = sps * self.stream.src_Fs / self.state.Fs
        self.state.sps = sps
        self.op.sps = sps

    def state_of(self, stage):
        """Get (creating on first use) the carried state of a stage"""
        try:
            return self.stream.states[stage]
        except KeyError:
            st = self.stream.states[stage] = stage.new_state(self)
            return st

    def __str__(self):
        return f"Meta: Fs{self.state.Fs}, sps{self.state.sps}, cf{self.state.cf}"
