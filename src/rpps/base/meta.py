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
    busy: seconds each stage spent in `step`; air: seconds of signal the source produced. busy/air (`load`) over 1
        means that stage can't keep up in real time
    dropped: blocks a threaded source discarded because the stages fell behind (Blocks.threaded(drop=True))
    pos: items the source has produced so far: the absolute position of its next block
    t0: time [s] of the source's item 0, when the source knows it (a radio's timestamp); None otherwise
    out_pos: items each stage has produced so far in this stream: the absolute position of its next output block
    est: what sync stages estimated, by name ({"align": {...}}), for whoever wants to know (det stays Acquire's)
    control: the control path (rpps.profile.control.Control): fields header steps published, position anchors, and
        what a profile runner did with them (applied, re-run, dropped). None until a step publishes

    Threads (Blocks.threaded): each stage's state is used only by the thread running that stage. `det` and
    `src_sps` have one writer (Acquire); `gen` lets blocks already on their way pick up a new sps (Meta.refresh).
    """
    __slots__ = ("states", "det", "blocks", "src_Fs", "src_sps", "gen", "busy", "air", "dropped", "pos", "t0", "out_pos", "est", "control")
    def __init__(self, Fs=None):
        self.states = {}
        self.det = None
        self.blocks = 0 # Blocks the source has produced
        self.src_Fs = Fs # Rate of the source
        self.src_sps = None # sps at the source, once a stage has learned it (Acquire): the next blocks start from it
        self.gen = 0 # Bumped when src_sps changes
        self.busy = {}
        self.air = 0.0
        self.dropped = 0
        self.pos = 0
        self.t0 = None
        self.out_pos = {}
        self.est = {}
        self.control = None

    def load(self):
        """Fraction of real time each stage takes: {stage: busy/air}. Over 1.0, it is too slow for a live source"""
        return {st: t/self.air for st, t in self.busy.items()} if self.air > 0 else {}

    def restart(self):
        """Forget everything: the next block starts a new stream"""
        self.states.clear()
        self.det = None
        self.blocks = 0
        self.src_sps = None
        self.gen += 1
        self.busy = {}
        self.air = 0.0
        self.dropped = 0
        self.pos = 0
        self.t0 = None
        self.out_pos = {}
        self.est = {}
        self.control = None

class Meta:
    """The rate information of one block of data, as a stage sees it

        Fs, sps, cf, isps, Ts: the rates of the block coming in (read-only for stages)
        op: the rates of the block going out. A stage changes these (meta.op.Fs /= 2), the carrier applies them
        delay: group delay of the data so far [s]. A stage adds its own
        stream: the Stream this block belongs to (stage state, detection)
        pos: absolute position of the block's first item in the stream, counted in items at this block's rate. It does
            not depend on how the stream was cut into blocks, so estimators can place their windows by it
        flush: this block ends the stream: a stage gives out what it was holding back
    """
    __slots__ = ("state", "op", "delay", "stream", "gen", "pos", "flush")
    def __init__(self, Fs=1.0, sps=0.0, cf=0.0, delay=0.0, stream=None, pos=0):
        self.stream = Stream(Fs) if stream is None else stream
        if self.stream.src_Fs is None:
            self.stream.src_Fs = Fs
        if self.stream.src_sps is not None:
            sps = self.stream.src_sps # A stage already learned the source's sps
        self.state = Rate(Fs, sps, cf)
        self.op = Rate(Fs, sps, cf)
        self.delay = delay
        self.gen = self.stream.gen
        self.pos = int(pos)
        self.flush = False

    def time_of(self, k=0):
        """Source time [s] of item k of this block: the stream's t0, plus the position at this rate, less the group delay.
        None when the source gave no time anchor"""
        t0 = self.stream.t0
        return None if t0 is None else t0 + (self.pos + k)/self.state.Fs - self.delay

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
        m.gen = self.gen
        m.pos = self.pos
        m.flush = False
        return m

    def refresh(self):
        """Pick up an sps learned after this block left the source

        One thread: never needed, every block starts after the last one finished. Threaded, blocks already between
        the source and Acquire carry the old sps: scale the source's new one by the Fs ratio, as the stages would have.
        """
        s = self.stream
        if self.gen != s.gen:
            self.gen = s.gen
            if s.src_sps is not None and s.src_Fs:
                sps = s.src_sps * self.state.Fs / s.src_Fs
                self.state.sps = sps
                self.op.sps = sps

    def calibrate_sps(self, sps):
        """Declare the sps at the current stage of the chain

        Blocks come from the source with the source's sps, so the sps at the source is worked out through the
        Fs ratio (decimation/resampling scale both) and handed to the blocks that follow.
        """
        self.stream.src_sps = sps * self.stream.src_Fs / self.state.Fs
        self.stream.gen += 1
        self.gen = self.stream.gen
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
